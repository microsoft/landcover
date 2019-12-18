#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
'''This script takes an input raster file (e.g. the 2013 NLCD layer) and a target raster file (e.g. a small NAIP tile)
and creates an output file that contains a crop of the input raster file that has been reprojected to match the
spatial dimension and resolution of the target file.
'''
import sys
import os
import time
import collections
import subprocess
import argparse

import rasterio


def main():
    parser = argparse.ArgumentParser(description="Reprojection script")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--input_fn", action="store", dest="input_fn", type=str, help="Path to input raster (this is the larger file that we want to chunk up)", required=True)
    parser.add_argument("--target_fn", action="store", dest="target_fn", type=str, help="Path to target raster (this is the shape that we want to copy)", required=True)
    parser.add_argument("--output_fn", action="store", dest="output_fn", type=str, help="Path to output", required=True)
   
    args = parser.parse_args(sys.argv[1:])

    start_time = float(time.time())

    if args.verbose:
        print("Starting reprojection")
        print("--input_fn: %s" % (args.input_fn))
        print("--target_fn: %s" % (args.target_fn))
        print("--output_fn: %s" % (args.output_fn))
        print("")

        print("Reading metadata from target_fn")
    f = rasterio.open(args.target_fn,"r")
    left, bottom, right, top = f.bounds
    crs = f.crs.to_string()
    height, width = f.height, f.width
    f.close()

    assert crs.startswith("EPSG")

    command = [
        "gdalwarp",
        "" if args.verbose else "-q",
        "-overwrite",
        "-ot", "Byte",
        "-t_srs", crs,
        "-r", "near",
        "-of", "GTiff",
        "-te", str(left), str(bottom), str(right), str(top),
        "-ts", str(width), str(height),
        "-co", "COMPRESS=LZW",
        "-co", "BIGTIFF=YES",
        args.input_fn,
        args.output_fn
    ]
    if args.verbose:
        print("Reprojecting input_fn to shape of target_fn")
    subprocess.call(command)

    if args.verbose:
        print("Finished in %0.4f seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
