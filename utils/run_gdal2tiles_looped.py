import sys
import os
import time
import subprocess

NUM_WORKERS = 12
INPUT_FILE = "/home/caleb/landcover/data/imagery/Site5_120.tif"
OUTPUT_TILE_DIR = "/home/caleb/landcover/data/basemaps/Site5_120_tiles/"

for zoom_level in range(8,19):
   print("Running zoom level %d" % (zoom_level))
   command = [
       "gdal2tiles.py", "-z", str(zoom_level), "--processes=%d" % (NUM_WORKERS), INPUT_FILE, OUTPUT_TILE_DIR
   ]
   subprocess.call(" ".join(command), shell=True)