import sys
import os
import time

import numpy as np
from enum import Enum

from urllib.request import urlopen

import fiona
import fiona.transform
import fiona.crs

import shapely
import shapely.geometry

import rasterio
import rasterio.warp
import rasterio.crs
import rasterio.io
import rasterio.mask
import rasterio.transform
import rasterio.merge
\
import rtree

import mercantile

import cv2
import pickle
import glob

import GeoTools

from web_tool.frontend_server import ROOT_DIR


class GeoDataTypes(Enum):
    NAIP = 1
    NLCD = 2
    LANDSAT_LEAFON = 3
    LANDSAT_LEAFOFF = 4
    BUILDINGS = 5
    LANDCOVER = 6

# ------------------------------------------------------------------------------
# Caleb's methods for finding which NAIP tiles are assosciated with an input extent
# 
# TODO: Assume that the tile_index.dat file is already created, make a separate script
# for generating it
# ------------------------------------------------------------------------------

assert all([os.path.exists(fn) for fn in [
    ROOT_DIR + "/data/tile_index.dat",
    ROOT_DIR + "/data/tile_index.idx",
    ROOT_DIR + "/data/tiles.p"
]])
TILES = pickle.load(open(ROOT_DIR + "/data/tiles.p", "rb"))

with fiona.open("/mnt/afs/chesapeake/landcover/data/yangon.geojson") as f:
    yangon_outline = next(iter(f))
    yangon_outline = shapely.geometry.shape(yangon_outline["geometry"])

with fiona.open("data/HCMC_outline.geojson") as f:
    hcmc_outline = next(iter(f))
    hcmc_outline = shapely.geometry.shape(hcmc_outline["geometry"])


def lookup_tile_by_geom(extent):
    tile_index = rtree.index.Index(ROOT_DIR + "/data/tile_index")

    geom = GeoTools.extent_to_transformed_geom(extent, "EPSG:4269")
    
    # Add some margin
    #minx, miny, maxx, maxy = shape(geom).buffer(50).bounds
    minx, miny, maxx, maxy = shapely.geometry.shape(geom).bounds
    geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))

    geom = shapely.geometry.shape(geom)
    intersected_indices = list(tile_index.intersection(geom.bounds))
    for idx in intersected_indices:
        intersected_fn = TILES[idx][0]
        intersected_geom = TILES[idx][1]
        if intersected_geom.contains(geom):
            return intersected_fn

    if len(intersected_indices) > 0:
        raise ValueError("Error, there are overlaps with tile index, but no tile completely contains selection")
    else:

        geom = GeoTools.extent_to_transformed_geom(extent, "EPSG:4326")
        if yangon_outline.contains(shapely.geometry.shape(geom)):
            return "/mnt/afs/chesapeake/landcover/data/merged_rgbnir_byte.tif"
        elif hcmc_outline.contains(shapely.geometry.shape(geom)):
            return "data/ThuDuc_WGS84.tif"
        else:
            raise ValueError("No tile intersections")

# ------------------------------------------------------------------------------

def get_data_by_extent(naip_fn, extent, geo_data_type, padding=20):

    if geo_data_type == GeoDataTypes.NAIP:
        fn = naip_fn
    elif geo_data_type == GeoDataTypes.NLCD:
        fn = naip_fn.replace("/esri-naip/", "/resampled-nlcd/")[:-4] + "_nlcd.tif"
    elif geo_data_type == GeoDataTypes.LANDSAT_LEAFON:
        fn = naip_fn.replace("/esri-naip/data/v1/", "/resampled-landsat8/data/leaf_on/")[:-4] + "_landsat.tif"
    elif geo_data_type == GeoDataTypes.LANDSAT_LEAFOFF:
        fn = naip_fn.replace("/esri-naip/data/v1/", "/resampled-landsat8/data/leaf_off/")[:-4] + "_landsat.tif"
    elif geo_data_type == GeoDataTypes.BUILDINGS:
        fn = naip_fn.replace("/esri-naip/", "/resampled-buildings/")[:-4] + "_building.tif"
    elif geo_data_type == GeoDataTypes.LANDCOVER:
        # TODO: Add existence check
        fn = naip_fn.replace("/esri-naip/", "/resampled-lc/")[:-4] + "_lc.tif"
    else:
        raise ValueError("GeoDataType not recognized")

    f = rasterio.open(fn, "r")
    src_index = f.index
    src_crs = f.crs
    transformed_geom = GeoTools.extent_to_transformed_geom(extent, f.crs.to_dict())
    transformed_geom = shapely.geometry.shape(transformed_geom)
    buffed_geom = transformed_geom.buffer(padding)
    geom = shapely.geometry.mapping(shapely.geometry.box(*buffed_geom.bounds))
    src_image, src_transform = rasterio.mask.mask(f, [geom], crop=True)
    f.close()

    return src_image, src_crs, src_transform, buffed_geom.bounds, src_index


def warp_data_to_3857(src_img, src_crs, src_transform, src_bounds):
    ''' Assume that src_img is (height, width, channels)
    '''
    assert len(src_img.shape) == 3
    src_height, src_width, num_channels = src_img.shape

    src_img_tmp = np.rollaxis(src_img.copy(), 2, 0)
    
    dst_crs = rasterio.crs.CRS.from_epsg(3857)
    dst_bounds = rasterio.warp.transform_bounds(src_crs, dst_crs, *src_bounds)
    dst_transform, width, height = rasterio.warp.calculate_default_transform(
        src_crs,
        dst_crs,
        width=src_width, height=src_height,
        left=src_bounds[0],
        bottom=src_bounds[1],
        right=src_bounds[2],
        top=src_bounds[3],
        resolution=1
    )

    dst_image = np.zeros((num_channels, height, width), np.float32)
    rasterio.warp.reproject(
        source=src_img_tmp,
        destination=dst_image,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=rasterio.warp.Resampling.nearest
    )
    dst_image = np.rollaxis(dst_image, 0, 3)
    
    return dst_image, dst_bounds


def crop_data_by_extent(src_img, src_bounds, extent):

    original_bounds = np.array((extent["xmin"], extent["ymin"], extent["xmax"], extent["ymax"]))
    new_bounds = np.array(src_bounds)

    diff = np.round(original_bounds - new_bounds).astype(int)
    print(diff)
    return src_img[diff[1]:diff[3], diff[0]:diff[2], :]


def get_image_by_xyz_from_url(tile):
    req = urlopen("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/%d/%d/%d" % (
        tile.z, tile.y, tile.x
    ))
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img    

def get_tile_as_virtual_raster(tile):
    img = get_image_by_xyz_from_url(tile)
    geom = shapely.geometry.shape(mercantile.feature(tile)["geometry"])
    minx, miny, maxx, maxy = geom.bounds
    dst_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, 256, 256)
    dst_profile = {
        "driver": "GTiff",
        "width": 256,
        "height": 256,
        "transform": dst_transform,
        "crs": "epsg:4326",
        "count": 3,
        "dtype": "uint8"
    }    
    test_f = rasterio.io.MemoryFile()
    with test_f.open(**dst_profile) as test_d:
        test_d.write(img[:,:,0], 1)
        test_d.write(img[:,:,1], 2)
        test_d.write(img[:,:,2], 3)
    test_f.seek(0)
    
    return test_f

def get_esri_by_extent(extent, padding=0.0001, zoom_level=17):
    transformed_geom = GeoTools.extent_to_transformed_geom(extent, "epsg:4326")
    transformed_geom = shapely.geometry.shape(transformed_geom)
    buffed_geom = transformed_geom.buffer(padding)
    
    minx, miny, maxx, maxy = buffed_geom.bounds
    
    virtual_files = []
    virtual_datasets = []
    for i, tile in enumerate(mercantile.tiles(minx, miny, maxx, maxy, zoom_level)):
        f = get_tile_as_virtual_raster(tile)
        virtual_files.append(f)
        virtual_datasets.append(f.open())
    out_image, out_transform = rasterio.merge.merge(virtual_datasets, bounds=(minx, miny, maxx, maxy))

    for ds in virtual_datasets:
        ds.close()
    for f in virtual_files:
        f.close()
    
    dst_crs = rasterio.crs.CRS.from_epsg(4326)
    dst_profile = {
        "driver": "GTiff",
        "width": out_image.shape[1],
        "height": out_image.shape[0],
        "transform": out_transform,
        "crs": "epsg:4326",
        "count": 3,
        "dtype": "uint8"
    }
    test_f = rasterio.io.MemoryFile()
    with test_f.open(**dst_profile) as test_d:
        test_d.write(out_image[:,:,0], 1)
        test_d.write(out_image[:,:,1], 2)
        test_d.write(out_image[:,:,2], 3)
    test_f.seek(0)
    with test_f.open() as test_d:
        dst_index = test_d.index
    test_f.close()

    r,g,b = out_image
    out_image = np.stack([r,g,b,r])
    
    return out_image, dst_crs, out_transform, (minx, miny, maxx, maxy), dst_index