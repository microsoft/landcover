import sys
import os
import time
import string

import numpy as np
from collections import defaultdict

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

def naip2id(naip):
    return '_'.join(naip.split('/')[-1].split('_')[:-1])

def get_naip_same_loc(naip):
    if naip2id(naip) in naip_d:
        return naip_d[naip2id(naip)]
    return [naip,]

def naip_fn_to_pred_tile_fn(naip_fn):
    return os.path.basename(naip_fn).split('.mrf')[0] + '_tilepred'

def get_landsat_by_extent(naip_fn, extent, pad_rad):
    ls_on_fn = naip_fn.replace("/esri-naip/data/v1/", "/resampled-landsat8/data/leaf_on/")[:-4] + "_landsat.tif"
    ls_off_fn = naip_fn.replace("/esri-naip/data/v1/", "/resampled-landsat8/data/leaf_off/")[:-4] + "_landsat.tif"
    #print(ls_on_fn)
    #print(ls_off_fn)

    f = rasterio.open(ls_on_fn, "r")
    geom = extent_to_transformed_geom(extent, f.crs["init"])
    minx, miny, maxx, maxy = shapely.geometry.shape(geom).buffer(pad_rad).bounds
    geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
    out_image1, out_transform = rasterio.mask.mask(f, [geom], crop=True)
    #raster_mask, _, win = rasterio.mask.raster_geometry_mask(f, [geom], crop=True)
    f.close()

    f = rasterio.open(ls_off_fn, "r")
    geom = extent_to_transformed_geom(extent, f.crs["init"])
    minx, miny, maxx, maxy = shapely.geometry.shape(geom).buffer(pad_rad).bounds
    geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
    out_image2, out_transform = rasterio.mask.mask(f, [geom], crop=True)
    f.close()

    return np.concatenate((out_image1, out_image2), axis=0)

def get_blg_by_extent(naip_fn, extent, pad_rad):
    blg_fn = naip_fn.replace("/esri-naip/data/v1/", "/resampled-buildings/data/v1/")[:-4] + "_building.tif"

    f = rasterio.open(blg_fn, "r")
    geom = extent_to_transformed_geom(extent, f.crs["init"])
    minx, miny, maxx, maxy = shapely.geometry.shape(geom).buffer(pad_rad).bounds
    geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
    out_image, out_transform = rasterio.mask.mask(f, [geom], crop=True)
    f.close()

    return out_image

def get_nlcd_by_extent(naip_fn, extent, pad_rad):
    nlcd_fn = naip_fn.replace("/esri-naip/", "/resampled-nlcd/")[:-4] + "_nlcd.tif"

    f = rasterio.open(nlcd_fn, "r")
    geom = extent_to_transformed_geom(extent, f.crs["init"])
    minx, miny, maxx, maxy = shapely.geometry.shape(geom).buffer(pad_rad).bounds
    geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
    out_image, out_transform = rasterio.mask.mask(f, [geom], crop=True)
    f.close()

    out_image = np.squeeze(out_image)
    out_image = np.vectorize(cid.__getitem__)(out_image)

    return out_image

def get_lc_by_extent(naip_fn, extent, pad_rad):
    naip_fns = get_naip_same_loc(naip_fn)
    for naip_fname in naip_fns:
        lc_fn = naip_fname.replace("/esri-naip/", "/resampled-lc/")[:-4] + "_lc.tif"
        if os.path.exists(lc_fn):
            #print(lc_fn)

            f = rasterio.open(lc_fn, "r")
            geom = extent_to_transformed_geom(extent, f.crs["init"])
            minx, miny, maxx, maxy = shapely.geometry.shape(geom).buffer(pad_rad).bounds
            geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
            out_image, out_transform = rasterio.mask.mask(f, [geom], crop=True)
            f.close()

            out_image = np.squeeze(out_image)
            out_image[out_image>=7] = 0

            return out_image

    return np.zeros((10, 10), dtype=np.float32)

def get_naip_by_extent(naip_fn, extent):
    naip_fns = get_naip_same_loc(naip_fn)
    naip_fn = naip_fns[-1]
    #print(naip_fn)

    f = rasterio.open(naip_fn, "r")
    geom = extent_to_transformed_geom(extent, f.crs["init"])

    pad_rad = 4
    bounds = shapely.geometry.shape(geom).buffer(pad_rad).bounds
    minx, miny, maxx, maxy = bounds
    while (maxx-minx) < 280 or (maxy-miny) < 280:
        pad_rad += 20
        bounds = shapely.geometry.shape(geom).buffer(pad_rad).bounds
        minx, miny, maxx, maxy = bounds
    
    geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))

    out_image, out_transform = rasterio.mask.mask(f, [geom], crop=True)
    crs = f.crs.copy()
    f.close()

    return out_image, out_transform, crs, bounds, pad_rad

def get_naip_by_extent_fixed(naip_fn, extent):
    naip_fns = get_naip_same_loc(naip_fn)
    naip_fn = naip_fns[-1]
    #print(naip_fn)

    
    f = rasterio.open(naip_fn, "r")
    geom = extent_to_transformed_geom(extent, f.crs["init"])
    pad_rad = 15
    buffed_geom = shapely.geometry.shape(geom).buffer(pad_rad)
    minx, miny, maxx, maxy = buffed_geom.bounds
    geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
    out_image, out_transform = rasterio.mask.mask(f, [geom], crop=True)
    src_crs = f.crs.copy()
    f.close()
    
    
    dst_crs = {"init": "EPSG:3857"}
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
    pad = int(np.round((dst_image.shape[1] - w) / 2))

    return dst_image, dst_transform, src_crs, 0, pad
    
    #return out_image, out_transform, src_crs, 0, 0

def lookup_tile_by_geom(geom):
    tile_index = rtree.index.Index("data/tile_index")

    # Add some margin
    #minx, miny, maxx, maxy = shape(geom).buffer(50).bounds
    minx, miny, maxx, maxy = shapely.geometry.shape(geom).bounds
    geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))

    geom = shapely.geometry.shape(geom)
    intersected_indices = list(tile_index.intersection(geom.bounds))
    for idx in intersected_indices:
        intersected_fn = tiles[idx][0]
        intersected_geom = tiles[idx][1]
        if intersected_geom.contains(geom):
            return True, intersected_fn

    if len(intersected_indices) > 0:
        # we picked something that overlaps the border between tiles but isn't totally in one
        msg = "Error, there are overlaps with tile index, but no tile completely contains selection"
        print(msg)
        return False, msg
    else:
        # we didn't find anything
        msg = "No tile intersections"
        print(msg)
        return False, msg

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
        intersected_fn = tiles[idx][0]
        intersected_geom = tiles[idx][1]
        geom = shapely.geometry.mapping(intersected_geom)
        geom = fiona.transform.transform_geom("EPSG:4269", "EPSG:3857", geom)
        return geom, intersected_fn

    print('No tile intersecton')
    return None, None

def extent_to_transformed_geom(extent, dest_crs="EPSG:4269"):
    left, right = extent["xmin"], extent["xmax"]
    top, bottom = extent["ymax"], extent["ymin"]

    geom = {
        "type": "Polygon",
        "coordinates": [[(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]]
    }

    # The map navigator uses EPSG:3857 and Caleb's indices use EPSG:4269
    return fiona.transform.transform_geom("EPSG:3857", dest_crs, geom)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

NLCD_CLASSES = [
    0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95, 255
]
cid = defaultdict(lambda: 0, {cl:i for i,cl in enumerate(NLCD_CLASSES)})

if not os.path.exists("data/tile_index.dat"):
    #------------------------------------
    # We need to load the shapefile and create a tile index for quick point queries
    #------------------------------------
    tic = float(time.time())
    print("Creating spatial index and pickling tile name dictionary")
    tile_index = rtree.index.Index("data/tile_index")
    tiles = {}
    f = fiona.open("data/best_tiles.shp", "r")
    count = 0
    for feature in f:
        if count % 10000 == 0:
            print("Loaded %d shapes..." % (count))

        fid = feature["properties"]["location"]
        geom = shapely.geometry.shape(feature["geometry"])
        tile_index.insert(count, geom.bounds)
        tiles[count] = (fid, geom)

        count += 1
    f.close()
    tile_index.close()

    pickle.dump(tiles, open("data/tiles.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print("Finished creating spatial index in %0.4f seconds" % (time.time() - tic))
else:
    #------------------------------------
    # Tile index already exists
    #------------------------------------
    print("Spatial index already exists, loading")
    tiles = pickle.load(open("data/tiles.p", "rb"))

print("Spatial index loaded")

# Load naip dictionary
naip_d = {}
fdid = open('data/list_all_naip.txt')
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