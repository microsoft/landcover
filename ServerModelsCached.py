import sys, os, time

import numpy as np

import fiona
import fiona.transform
import shapely
import shapely.geometry
import rasterio
import rasterio.mask

def extent_to_geom(extent, dest_crs="EPSG:4269"):
    left, right = extent["xmin"], extent["xmax"]
    top, bottom = extent["ymax"], extent["ymin"]

    geom = {
        "type": "Polygon",
        "coordinates": [[(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]]
    }

    src_crs = "EPSG:" + str(extent["spatialReference"]["latestWkid"])

    return fiona.transform.transform_geom(src_crs, dest_crs, geom)

def run(naip, fn, extent, buffer):
    return get_cached_by_extent(fn, extent, buffer)

def get_cached_by_extent(fn, extent, buffer):
    fn = fn.replace("esri-naip/", "full-usa-output/7_10_2018/")[:-4] + "_prob.tif"
    f = rasterio.open(fn, "r")
    geom = extent_to_geom(extent, f.crs["init"])
    minx, miny, maxx, maxy = shapely.geometry.shape(geom).buffer(buffer).bounds
    geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
    out_image, out_transform = rasterio.mask.mask(f, [geom], crop=True)
    f.close()
    print(out_image.shape)
    out_image = np.rollaxis(out_image, 0, 3)
    return out_image / 255.0, "Full-US-prerun"

