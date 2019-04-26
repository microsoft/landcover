import argparse
import numpy as np
import json
from collections import defaultdict

import pdb

import torch
import torch.nn as nn
from training.pytorch.models.unet import Unet
from training.pytorch.utils.eval_segm import mean_IoU
from training.pytorch.data_loader import DataGenerator
from torch.utils import data

import rasterio
import rasterio.mask
import shapely
import shapely.geometry

parser = argparse.ArgumentParser()

parser.add_argument('--test_tile_fn', type=str, help="Filename with tile file names in npy format", default="training/data/finetuning/test1.txt")

args = parser.parse_args()


NLCD_CLASSES = [
        0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95, 255
    ]
NLCD_CLASSES_TO_IDX = defaultdict(lambda: 0, {cl:i for i,cl in enumerate(NLCD_CLASSES)})
NLCD_CLASS_IDX = range(len(NLCD_CLASSES))


def bounds_intersection(bound0, bound1):
    left0, bottom0, right0, top0 = bound0
    left1, bottom1, right1, top1 = bound1
    left, bottom, right, top = \
        max([left0, left1]), max([bottom0, bottom1]), \
        min([right0, right1]), min([top0, top1])
    return (left, bottom, right, top)


def main():
    f = open(args.test_tile_fn, "r")
    test_tiles_files = f.read().strip().split("\n")
    f.close()

    for count, naip_fn in enumerate(test_tiles_files):
        print(count, len(test_tiles_files))

        lc_fn = naip_fn.replace("esri-naip", "resampled-lc")[:-4] + "_lc.tif"
        nlcd_fn = naip_fn.replace("esri-naip", "resampled-nlcd")[:-4] + "_nlcd.tif"

        naip_f = rasterio.open(naip_fn, "r")
        naip_bounds = naip_f.bounds

        lc_f = rasterio.open(lc_fn, "r")
        lc_bounds = lc_f.bounds

        nlcd_f = rasterio.open(nlcd_fn, "r")
        nlcd_bounds = nlcd_f.bounds

        bounds = bounds_intersection(bounds_intersection(naip_bounds, lc_bounds), nlcd_bounds)
        left, bottom, right, top = bounds
        geom = shapely.geometry.mapping(shapely.geometry.box(left, bottom, right, top, ccw=True))

        naip_data, _ = rasterio.mask.mask(naip_f, [geom], crop=True)
        #naip_data = np.rollaxis(naip_data, 0, 3)
        naip_f.close()
        lc_data, _ = rasterio.mask.mask(lc_f, [geom], crop=True)
        #lc_data = np.squeeze(lc_data)
        lc_f.close()
        nlcd_data, _ = rasterio.mask.mask(nlcd_f, [geom], crop=True)
        nlcd_f.close()
        nlcd_data = np.vectorize(NLCD_CLASSES_TO_IDX.__getitem__)(nlcd_data).astype(np.uint8)

        #print(naip_fn, naip_data.shape, naip_data.dtype)
        #print(nlcd_fn, nlcd_data.shape, nlcd_data.dtype)
        #print(lc_fn, lc_data.shape, lc_data.dtype)

        _, height, width = naip_data.shape

        merged = np.concatenate([
            naip_data.astype(np.float32) / 255.0,
            lc_data,
            nlcd_data,
        ])

        lc_string = '_'.join(map(str,get_lc_stats(merged[4,:,:])))
        nlcd_string = '_'.join(map(str,get_nlcd_stats(merged[5:,:])))
        
        output_fn = tile_file_name.replace('.mrf', '.npy')
        
        np.save(output_fn, merged[np.newaxis].data)


if __name__ == '__main__':
    main()
