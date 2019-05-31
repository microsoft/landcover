#!/usr/bin/env python

"""
Downloads images + metadata from Planet

This gives some examples of downloading information from the Ayerawady region
collected over the last year.
"""
from argparse import ArgumentParser
import glob
import json
import numpy as np
import os.path
import scipy.sparse
import sys
import time
sys.path.append("../../..")

import training.pytorch.expers.satellite as sat
import training.pytorch.expers.osm as osm

parser = ArgumentParser()
parser.add_argument("-a", "--api-key", dest="api_key")
parser.add_argument("-d", "--directory", dest="directory", default=".")
args = parser.parse_args([])

assets = [
    {"item_type": "PSScene4Band", "id": "20170423_125001_0c81"}
]

# download images we requested
sat.parallel_downloads(assets, args.api_key, out_dir=args.directory)

for xml_path in glob.glob("{}/*.xml".format(args.directory)):
    # parse the XML
    base = os.path.basename(xml_path)
    parsed_data = sat.planet_xml(xml_path, os.path.basename(xml_path)[:2].lower())
    path = xml_path.replace(".xml", "")
    with open("{}.json".format(path), "w") as f:
        parsed_data["time"] = parsed_data["time"].strftime("%Y-%m-%d %H:%M:%S")
        json.dump(parsed_data, f)

    # get the OSM for this region
    polygon = list(parsed_data["region"]["corners"].values())
    osm.write_geojson(polygon, path)
    geojson = json.load(open(path + ".geojson", "r"))

    # save the raster
    img = osm.make_image(geojson, np.array(polygon), parsed_data["dimension"][:2])
    img = scipy.sparse.csc_matrix(img)
    scipy.sparse.save_npz("{}_mask.npz".format(path), img)


# look at the plots, if you want
#
# import matplotlib.pyplot as plt
# import cv2
#
# i = 0
# fname = "{}/{}_{}".format(args.directory, assets[i]["item_type"], assets[i]["id"])
# plt.imshow(cv2.imread("{}.tiff".format(fname)), alpha=0.9)
# plt.imshow(scipy.sparse.load_npz("{}_mask.npz".format(fname)).todense(), alpha=0.3)
# plt.show()