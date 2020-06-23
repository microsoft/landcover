#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Caleb Robinson <calebrob6@gmail.com>
#
'''Script for creating the shapefile used in `create_spatial_index.py`
'''
import sys, os, time
import subprocess

def main():

    if len(sys.argv) < 3:
        print("Usage: ./create_shapefile_from_files.py PATH_TO_FILELIST PATH_TO_OUTPUT_SHAPEFILE")
        return

    with open(sys.argv[1], "r") as f:
        fns = f.read().strip().split("\n")

    print("Found %d files" % (len(fns)))

    tic = float(time.time())
    for i, fn in enumerate(fns):
        if i % 1000 == 0:
            print("%d/%d\t%0.4f%%\t%0.4f seconds" % (i, len(fns), i/len(fns)*100, time.time()-tic))
            tic = float(time.time())

        command = [
            "gdaltindex",
            "-t_srs", "epsg:4326",
            sys.argv[2], fn
        ]
        subprocess.call(command)



if __name__ == "__main__":
    main()