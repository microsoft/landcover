import sys
import os
import time

import numpy as np
from enum import Enum

import fiona
import fiona.transform
import shapely
import shapely.geometry
import rasterio
import rasterio.mask
import rasterio.merge
import rasterio.warp
import rtree

import cv2
import pickle
import glob

import GeoTools

class GeoDataTypes(Enum):
    NAIP = 1
    NLCD = 2
    LANDSAT_LEAFON = 3
    LANDSAT_LEAFOFF = 4
    BUILDINGS = 5
    LANDCOVER = 6

# ------------------------------------------------------------------------------
# Le's methods for finding which NAIP tiles exist for a given "filename id"
#
# E.g. given "m_3907638_nw_18_1_20150815.mrf", find all files from other years that
# match "m_3907638_nw_18_1*"
# 
# TODO: Make this lookup faster, currently O(n) where n is total number of tiles 100k-1000k
# ------------------------------------------------------------------------------

def naip2id(naip):
    return '_'.join(naip.split('/')[-1].split('_')[:-1])

def get_naip_same_loc(naip):
    if naip2id(naip) in naip_d:
        return naip_d[naip2id(naip)]
    return [naip,]

assert all([os.path.exists(fn) for fn in [
    "data/list_all_naip.txt",
]])

naip_d = {}
fdid = open('data/list_all_naip.txt', 'r')
while True:
    line = fdid.readline().strip()
    if not line:
        break
    naipid = naip2id(line)
    if naipid in naip_d:
        naip_d[naipid] += [line,]
    else:
        naip_d[naipid] = [line,]
fdid.close()

# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Caleb's methods for finding which NAIP tiles are assosciated with an input extent
# 
# TODO: Assume that the tile_index.dat file is already created, make a separate script
# for generating it
# ------------------------------------------------------------------------------


assert all([os.path.exists(fn) for fn in [
    "data/tile_index.dat",
    "data/tile_index.idx",
    "data/tiles.p"
]])
TILES = pickle.load(open("data/tiles.p", "rb"))


def lookup_tile_by_geom(geom):
    tile_index = rtree.index.Index("data/tile_index")

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
        raise ValueError("No tile intersections")

# ------------------------------------------------------------------------------

def get_data_by_extent(naip_fn, extent, geo_data_type):

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
        fn = naip_fname.replace("/esri-naip/", "/resampled-lc/")[:-4] + "_lc.tif"
    else:
        raise ValueError("GeoDataType not recognized")

    f = rasterio.open(fn, "r")
    geom = GeoTools.extent_to_transformed_geom(extent, f.crs["init"])
    pad_rad = 15 # TODO: this might need to be changed for much larger inputs
    buffed_geom = shapely.geometry.shape(geom).buffer(pad_rad)
    minx, miny, maxx, maxy = buffed_geom.bounds
    geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
    out_image, out_transform = rasterio.mask.mask(f, [geom], crop=True)
    src_crs = f.crs.copy()
    f.close()
    
    dst_crs = {"init": "EPSG:%s" % (extent["spatialReference"]["latestWkid"])}
    dst_transform, width, height = rasterio.warp.calculate_default_transform(
        src_crs,
        dst_crs,
        width=out_image.shape[2], height=out_image.shape[1],
        left=buffed_geom.bounds[0],
        bottom=buffed_geom.bounds[1],
        right=buffed_geom.bounds[2],
        top=buffed_geom.bounds[3],
        resolution=1
    )

    dst_image = np.zeros((out_image.shape[0], height, width), np.uint8)
    rasterio.warp.reproject(
            source=out_image,
            destination=dst_image,
            src_transform=out_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=rasterio.warp.Resampling.nearest
    )
    
    # Calculate the correct padding
    w = extent["xmax"] - extent["xmin"]
    padding = int(np.round((dst_image.shape[1] - w) / 2))

    return dst_image, padding

# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# Le's methods for predicting entire tile worth of data
#
# NOTE: I have not refactored these --Caleb
# ------------------------------------------------------------------------------

def naip_fn_to_pred_tile_fn(naip_fn):
    return os.path.basename(naip_fn).split('.mrf')[0] + '_tilepred'

def find_tile_and_load_pred(centerp):
    geom, naip_fn = center_to_tile_geom(centerp)
    naip_key = naip_fn_to_pred_tile_fn(naip_fn)
    fnames = []
    usernames = []
    for paths in glob.glob('img/*/{}_geotif.tif'.format(naip_key)):
        fnames.append(paths[len('img/'):-len('_geotif.tif')])
        usernames.append(os.path.basename(os.path.dirname(paths)))
    return fnames, usernames

def find_tile_and_save_pred(centerp, tif_names, username):
    datasets = [rasterio.open(fn, "r") for fn in tif_names[::-1]]

    # Get geom and CRS
    # Assumes the CRS of the tile is the same as the CRSes of patches
    geom, naip_fn = center_to_tile_geom(centerp)
    fid = rasterio.open(naip_fn, "r")
    tile_crs = fid.crs.copy()
    fid.close()
    geom = fiona.transform.transform_geom("EPSG:3857", tile_crs["init"], geom)

    # Merge all geospatial patches
    tile, tile_transform = rasterio.merge.merge(datasets, shapely.geometry.shape(geom).bounds, nodata=0)
    for fid in datasets:
        fid.close()

    # Save the resulting tile
    if not os.path.exists('img/{}'.format(username)):
        os.makedirs('img/{}'.format(username))
    fname_tif = 'img/{}/{}_geotif.tif'.format(username, naip_fn_to_pred_tile_fn(naip_fn))
    fid = rasterio.open(fname_tif, 'w', driver='GTiff',
            width=tile.shape[2], height=tile.shape[1], count=tile.shape[0], dtype=np.uint8,
            transform=tile_transform, crs=tile_crs, nodata=0)
    for ch in range(tile.shape[0]):
        fid.write(tile[ch, ...], ch+1)
    fid.close()

    # Get pngs in CRS EPSG3857 for display purpose
    dst_CRS = "EPSG:3857"

    png_tile_bounds = shapely.geometry.shape(geom).bounds
    dest_transform, width, height = rasterio.warp.calculate_default_transform(
            tile_crs, rasterio.crs.CRS({"init": dst_CRS}),
            width=tile.shape[2], height=tile.shape[1],
            left=png_tile_bounds[0], bottom=png_tile_bounds[1],
            right=png_tile_bounds[2], top=png_tile_bounds[3])

    tile_dest = np.zeros((tile.shape[0], height, width), np.uint8)
    rasterio.warp.reproject(
            source=tile,
            destination=tile_dest,
            src_transform=tile_transform,
            src_crs=tile_crs,
            dst_transform=dest_transform,
            dst_crs=rasterio.crs.CRS({"init": dst_CRS}),
            resampling=rasterio.warp.Resampling.nearest
    )
    tile_dest = np.swapaxes(tile_dest, 0, 1)
    tile_dest = np.swapaxes(tile_dest, 1, 2)
    mask = np.max(tile_dest, axis=2, keepdims=True) > 0
    tile_dest = tile_dest.astype(np.float32) / 255.0
    im_soft = np.round(255*pic(tile_dest, hard=False)).astype(np.uint8)
    im_hard = np.round(255*pic(tile_dest, hard=True)).astype(np.uint8) * mask
    fname_png = 'img/{}/{}_soft.png'.format(username, naip_fn_to_pred_tile_fn(naip_fn))
    cv2.imwrite(fname_png, cv2.cvtColor(im_soft, cv2.COLOR_RGB2BGR))
    fname_png = 'img/{}/{}_hard.png'.format(username, naip_fn_to_pred_tile_fn(naip_fn))
    cv2.imwrite(fname_png, cv2.cvtColor(im_hard, cv2.COLOR_RGB2BGR))

    return fname_tif

def center_to_tile_geom(centerp):
    xctr = centerp["xcenter"]
    yctr = centerp["ycenter"]

    geom = {
        "type": "Polygon",
        "coordinates": [[(xctr-1, yctr-1), (xctr+1, yctr-1), (xctr+1, yctr+1), (xctr-1, yctr+1), (xctr-1, yctr-1)]]
    }

    # The map navigator uses EPSG:3857 and Caleb's indices use EPSG:4269
    geom = fiona.transform.transform_geom("EPSG:3857", "EPSG:4269", geom)
    geom = shapely.geometry.shape(geom)

    tile_index = rtree.index.Index("data/tile_index")
    intersected_indices = list(tile_index.intersection(geom.bounds))
    for idx in intersected_indices:
        intersected_fn = TILES[idx][0]
        intersected_geom = TILES[idx][1]
        geom = shapely.geometry.mapping(intersected_geom)
        geom = fiona.transform.transform_geom("EPSG:4269", "EPSG:3857", geom)
        return geom, intersected_fn

    print('No tile intersecton')
    return None, None

# ------------------------------------------------------------------------------