#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Caleb Robinson <calebrob6@gmail.com>
#
'''Script for converting a list of files to COG format.

The input files can be of any type that GDAL can read.

If you select "--overwrite" then your input files must be TIFs (i.e. end with {".tif", ".tiff", ".TIF", ".TIFF"}). These files will be converted to COG and overwritten in place. You might lose data if the program crashes.
If you select "--suffix" with a suffix of "_SUFFIX" then for each input file "/PATH/INPUT.EXT", a COG file "/PATH/INPUT_SUFFIX.tif" will be written.

The files are converted using `gdal_translate`. Note: you must have GDAL version >=3.1 for this to work -- the script attempts to check for this.
You can install GDAL>3.1 with `conda create --name myenv "gdal>=3.1" -c conda-forge`.

The script writes a log file to the filename given by `--log_fn` that shows the return code of `gdal_translate` for each input file. A value of "0" means that everything went as expected.
'''
import sys, os, time
import argparse
import subprocess
from multiprocessing import Pool

import gdal

parser = argparse.ArgumentParser(description="Tile index creation script")

parser.add_argument("--input_fn", action="store", type=str, help="Path to file containing a list of files to convert", required=True)
parser.add_argument("--log_fn", action="store", type=str, help="Path to output logfile", default="cogify_log.csv")
parser.add_argument("--num_processes", action="store", type=int, help="Number of threads to use", default=12)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--overwrite", action="store_true", help="Overwrite existing files in place (only works if they all end with .tif or .tiff)")
group.add_argument("--suffix", action="store", help="Suffix to append to each file that we convert (appended before the extension)")

args = parser.parse_args(sys.argv[1:])

def do_work(fn):
   
    if args.overwrite:
        output_fn = fn
    else:
        directory = os.path.dirname(fn)
        filename = os.path.basename(fn)
        parts = filename.split(".")
        extension = parts[-1]
        filename = ".".join(parts[:-1])

        output_fn = os.path.join(directory, filename + args.suffix + ".tif")

    command = [
        "gdal_translate",
        "-co", "NUM_THREADS=ALL_CPUS",
        "-co", "BIGTIFF=YES",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-of", "COG",
        fn,
        output_fn
    ]

    result = subprocess.call(" ".join(command), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return (output_fn,result)


def main():

    assert os.path.exists(args.input_fn)
    version = gdal.VersionInfo()
    assert int(version[0:2]) >= 30 and int(version[2:4]) >= 10, "You must have GDAL version >= 3.1 to write COGS"

    #-----------------------------------
    with open(args.input_fn, "r") as f:
        fns = f.read().strip().split("\n")
    print("Found %d files" % (len(fns)))

    if args.overwrite:
        for fn in fns:
            assert fn.lower().endswith(".tif") or fn.lower().endswith(".tiff"), "If you want to overwrite files in place, then they all must be in TIF format (as we are converting to COG -- only makes sense if the file type is TIF)"
    else:
        assert args.suffix != "", "The suffix can't be an empty string because you might lose data."

    fns = fns[:4]

    p = Pool(args.num_processes)
    results = p.map(do_work, fns)

    with open(args.log_fn, "w") as f:
        f.write("input_fn,output_fn,result\n")
        for i in range(len(fns)):
            f.write("%s,%s,%d\n" % (fns[i], results[i][0], results[i][1]))

if __name__ == "__main__":
    main()
