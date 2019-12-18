#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
'''This script calculates the intersection of two input rasters, crops both rasters to the intersection, and saves
the results for each into new GeoTIFFs.

WARNING: This script assumes that both input rasters are in the same coordinate system and that there exists an intersection.
WARNING: The intersection is computed by diffing the bounds metadata from the files. 
'''
import sys
import os
import time
import collections

import argparse

import numpy as np

import rasterio
import rasterio.mask

import shapely
import shapely.geometry

def bounds_intersection(bound1, bound2):
    left1, bottom1, right1, top1 = bound1
    left2, bottom2, right2, top2 = bound2
    left, bottom, right, top = max([left1, left2]), max([bottom1, bottom2]), min([right1, right2]), min([top1, top2])
    return left, bottom, right, top

def write_new_tiff(fn, data, transform, crs):
    count, height, width = data.shape
    
    new_profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "crs": crs,
        "dtype": "uint8",
        "count": count,
        "transform": transform,
        "compress": "lzw"
    }
    
    f = rasterio.open(fn, "w", **new_profile)
    f.write(data)
    f.close()

def main():
    parser = argparse.ArgumentParser(description="Intersection script")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--input_fn1", action="store", dest="input_fn1", type=str, help="Path to first input raster", required=True)
    parser.add_argument("--input_fn2", action="store", dest="input_fn2", type=str, help="Path to second input raster", required=True)

    parser.add_argument("--output_fn1", action="store", dest="output_fn1", type=str, help="Path to first output GeoTIFF", required=True)
    parser.add_argument("--output_fn2", action="store", dest="output_fn2", type=str, help="Path to second output GeoTIFF", required=True)

    args = parser.parse_args(sys.argv[1:])
    start_time = float(time.time())

    assert os.path.exists(args.input_fn1)
    assert os.path.exists(args.input_fn2)
    assert not os.path.exists(args.output_fn1)
    assert not os.path.exists(args.output_fn2)
    
    f1 = rasterio.open(args.input_fn1, "r")
    crs = f1.crs["init"]
    f1_bounds = f1.bounds

    f2 = rasterio.open(args.input_fn2, "r")
    assert f2.crs == crs, "Files must have matching CRS"
    f2_bounds = f2.bounds

    difference = np.array(f1_bounds) - np.array(f2_bounds)
    if np.all(difference == 0):
        print("NOTE: The input files are already aligned")
        assert f1.width == f2.width
        assert f1.height == f2.height

    intersecting_bounds = bounds_intersection(f1_bounds, f2_bounds)
    intersecting_bounds_geom = shapely.geometry.mapping(shapely.geometry.box(*intersecting_bounds, ccw=True))

    data1, transform1 = rasterio.mask.mask(f1, [intersecting_bounds_geom], crop=True)
    data2, transform2 = rasterio.mask.mask(f2, [intersecting_bounds_geom], crop=True)
    f1.close()
    f2.close()

    write_new_tiff(args.output_fn1, data1, transform1, crs)
    write_new_tiff(args.output_fn2, data2, transform2, crs)
    
    print("Finished in %0.4f seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
