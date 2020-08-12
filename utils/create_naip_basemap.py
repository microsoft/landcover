'''Script for creating a XYZ style basemap for all NAIP imagery for a given (state, year).

This goes really fast on Azure VMs in US East with large number of cores.
'''
import sys
import os
import time
import subprocess
import tempfile
import urllib.request
from multiprocessing import Pool

import numpy as np

NAIP_BLOB_ROOT = 'https://naipblobs.blob.core.windows.net/naip'
temp_dir = os.path.join(tempfile.gettempdir(), 'naip')
os.makedirs(temp_dir, exist_ok=True)
NAIP_INDEX_FN = os.path.join(temp_dir, "naip_v002_index.csv")

OUTPUT_DIR = "/home/caleb/data/oh_2017_naip/"
OUTPUT_TILE_DIR = "/home/caleb/data/oh_2017_naip_tiles/"
NUM_WORKERS = 64
STATE = "oh" # use state code
YEAR = 2017
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_TILE_DIR, exist_ok=True)


def download_url(url, output_dir, force_download=False, verbose=False):
    """
    Download a URL
    """
    parsed_url = urllib.parse.urlparse(url)
    url_as_filename = os.path.basename(parsed_url.path)
    destination_filename = os.path.join(output_dir, url_as_filename)

    if (not force_download) and (os.path.isfile(destination_filename)):
        if verbose: print('Bypassing download of already-downloaded file {}'.format(os.path.basename(url)))
        return destination_filename
    
    if verbose: print('Downloading file {} to {}'.format(os.path.basename(url),destination_filename),end='')
    urllib.request.urlretrieve(url, destination_filename)  
    assert(os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    if verbose: print('...done, {} bytes.'.format(nBytes))
    return destination_filename

if not os.path.exists(NAIP_INDEX_FN):
    download_url("https://naipblobs.blob.core.windows.net/naip-index/naip_v002_index.csv", temp_dir)

fns = []
with open(NAIP_INDEX_FN, "r") as f:    
    for line in f:
        line = line.strip()
        if line != "":
            if line.endswith(".tif"):
                if ("/%s/" % (STATE)) in line and ("/%d/" % (YEAR)) in line:
                    fns.append(line)

print("Working on %d files" % (len(fns)))

def do_work(fn):
    time.sleep(np.random.random()*2)
    
    url = NAIP_BLOB_ROOT + "/" + fn
    output_fn = fn.split("/")[-1]
    output_tmp_fn = output_fn[:-4] + "_tmp.tif"
    
    command = [
        "GDAL_SKIP=DODS",
        "gdalwarp",
        "-t_srs", "epsg:3857",
        "'%s'" % (url),
        OUTPUT_DIR + output_tmp_fn
    ]
    subprocess.call(" ".join(command), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    
    command = [
        "gdal_translate",
        "-b", "1", "-b", "2", "-b", "3",
        OUTPUT_DIR + output_tmp_fn,
        OUTPUT_DIR + output_fn
    ]
    subprocess.call(" ".join(command), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    os.remove(OUTPUT_DIR + output_tmp_fn)

p = Pool(NUM_WORKERS)
_ = p.map(do_work, fns)



command = [
    "gdalbuildvrt", "-srcnodata", "\"0 0 0\"", "basemap.vrt", "%s*.tif" % (OUTPUT_DIR)
]
subprocess.call(" ".join(command), shell=True)


# We run gdal2tiles once for each zoom level that we want as output as the multithreaded part of gdal2tiles.py _only_ works for the largest zoom level you select.
# E.g. if we run `gdal2tiles.py -z 8-16 --processes=32 basemap.vrt OUTPUT_DIR/` then level 16 would be built with 32 threads, however levels 8 through 15 would be built with a single thread.
# This is OK if you are making a basemap for a relatively small area, however for large areas it is (much) faster to generate all the levels with multiple threads.  
for zoom_level in range(8,17):
   print("Running zoom level %d" % (zoom_level))
   command = [
       "gdal2tiles.py", "-z", str(zoom_level), "--processes=%d" % (NUM_WORKERS), "basemap.vrt", OUTPUT_TILE_DIR
   ]
   subprocess.call(" ".join(command), shell=True)


os.remove("basemap.vrt")