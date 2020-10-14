#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Caleb Robinson <calebrob6@gmail.com>
#
'''Script for converting a list of files to COG format.

The input files can be of any type that GDAL can read.

For "--suffix '_SUFFIX'", then for each input file "/PATH/INPUT.EXT", a COG file "/PATH/INPUT_SUFFIX.tif" will be written.

The files are converted using `gdal_translate`. Note: you must have GDAL version >=3.1 for this to work -- the script attempts to check for this.
You can install GDAL>3.1 with `conda create --name myenv "gdal>=3.1" -c conda-forge`.

The script writes a log file to the filename given by `--log_fn` that shows the return code of `gdal_translate` for each input file. A value of "0" means that everything went as expected.
'''
import sys, os, time
import argparse
import subprocess
from multiprocessing import Pool
import tempfile
import shutil

import gdal

parser = argparse.ArgumentParser(description="Tile index creation script")

parser.add_argument("--input_fn", action="store", type=str, help="Path to file containing a list of files to convert", required=True)
parser.add_argument("--log_fn", action="store", type=str, help="Path to output logfile", default="cogify_log.csv")
parser.add_argument("--num_processes", action="store", type=int, help="Number of threads to use", default=12)
parser.add_argument("--suffix", action="store", help="Suffix to append to each file that we convert (appended before the extension)", required=True)

args = parser.parse_args(sys.argv[1:])

def do_work(fn):

    directory = os.path.dirname(fn)
    filename = os.path.basename(fn)
    parts = filename.split(".")
    extension = parts[-1]
    filename = ".".join(parts[:-1])

    temp_fn = os.path.join(tempfile.gettempdir(), filename + args.suffix + ".tif")
    output_fn = os.path.join(directory, filename + args.suffix + ".tif")

    command = [
        "gdal_translate",
        #"-co", "NUM_THREADS=ALL_CPUS",
        "-co", "BIGTIFF=YES",
        "-co", "COMPRESS=LZW",
        "-co", "PREDICTOR=2",
        "-of", "COG",
        fn,
        temp_fn
    ]
    result = subprocess.call(" ".join(command), shell=True, stdout=subprocess.DEVNULL)
    if result == 0:
        shutil.copyfile(temp_fn, output_fn)
        os.remove(temp_fn)

    return (output_fn,result)


def main():

    assert os.path.exists(args.input_fn)
    version = gdal.VersionInfo()
    assert int(version[0:2]) >= 30 and int(version[2:4]) >= 10, "You must have GDAL version >= 3.1 to write COGS"

    #-----------------------------------
    with open(args.input_fn, "r") as f:
        fns = f.read().strip().split("\n")
    print("Found %d files" % (len(fns)))

    assert args.suffix != "", "The suffix can't be an empty string because you might lose data."

    p = Pool(args.num_processes)
    results = p.map(do_work, fns)

    with open(args.log_fn, "w") as f:
        f.write("input_fn,output_fn,result\n")
        for i in range(len(fns)):
            f.write("%s,%s,%d\n" % (fns[i], results[i][0], results[i][1]))

if __name__ == "__main__":
    main()
