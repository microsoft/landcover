import sys
import os
import time
import subprocess

NUM_WORKERS = 12
INPUT_FILE = sys.argv[1]
OUTPUT_TILE_DIR = sys.argv[2]

for zoom_level in range(8,20):
   print("Running zoom level %d" % (zoom_level))
   command = [
       "gdal2tiles.py", "-z", str(zoom_level), "--processes=%d" % (NUM_WORKERS), INPUT_FILE, OUTPUT_TILE_DIR
   ]
   subprocess.call(" ".join(command), shell=True)
