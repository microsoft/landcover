#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Caleb Robinson <calebrob6@gmail.com>
#
'''Script for creating the rtree spatial index used in `DataLoader.py`
'''
import sys
import os
import time

import fiona
import shapely.geometry
import rtree

import pickle

tic = float(time.time())
print("Creating spatial index and pickling tile name dictionary")
tile_index = rtree.index.Index("data/tile_index")
tiles = {}
f = fiona.open(sys.argv[1], "r")
count = 0
for feature in f:
    if count % 10000 == 0:
        print("Loaded %d shapes..." % (count))

    fid = feature["properties"]["fn"]
    geom = shapely.geometry.shape(feature["geometry"])
    tile_index.insert(count, geom.bounds)
    tiles[count] = (fid, geom)

    count += 1
f.close()
tile_index.close()

pickle.dump(tiles, open("data/tiles.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

print("Finished creating spatial index in %0.4f seconds" % (time.time() - tic))