#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: skip-file
#
# Caleb Robinson <calebrob6@gmail.com>
# and
# Le Hou <lehou0312@gmail.com>
"""Script for running a saved model file on a list of NAIP+Landsat images.
"""
# Stdlib imports
import sys
import os

# Here we look through the args to find which GPU we should use
# We must do this before importing keras, which is super hacky
# See: https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
# TODO: This _really_ should be part of the normal argparse code.
def parse_args(args, key):
    def is_int(s):
        try: 
            int(s)
            return True
        except ValueError:
            return False
    for i, arg in enumerate(args):
        if arg == key:
            if i+1 < len(sys.argv):
                if is_int(args[i+1]):
                    return args[i+1]
    return None
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID = parse_args(sys.argv, "--gpu")
if GPU_ID is not None: # if we passed `--gpu INT`, then set the flag, else don't
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import time
import subprocess
import datetime
import argparse

# Library imports
import numpy as np
import pandas as pd

import rasterio

import keras
import keras.models
import keras.metrics

def run_model_on_tile(model, naip_tile, inpt_size, output_size, batch_size):
    down_weight_padding = 40
    height = naip_tile.shape[0]
    width = naip_tile.shape[1]

    stride_x = inpt_size - down_weight_padding*2
    stride_y = inpt_size - down_weight_padding*2

    output = np.zeros((height, width, output_size), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
    kernel = np.ones((inpt_size, inpt_size), dtype=np.float32) * 0.1
    kernel[10:-10, 10:-10] = 1
    kernel[down_weight_padding:down_weight_padding+stride_y,
           down_weight_padding:down_weight_padding+stride_x] = 5

    batch = []
    batch_indices = []
    batch_count = 0

    for y_index in (list(range(0, height - inpt_size, stride_y)) + [height - inpt_size,]):
        for x_index in (list(range(0, width - inpt_size, stride_x)) + [width - inpt_size,]):
            naip_im = naip_tile[y_index:y_index+inpt_size, x_index:x_index+inpt_size, :]

            batch.append(naip_im)
            batch_indices.append((y_index, x_index))
            batch_count+=1

    model_output = model.predict(np.array(batch), batch_size=batch_size, verbose=0)
    for i, (y, x) in enumerate(batch_indices):
        output[y:y+inpt_size, x:x+inpt_size] += model_output[i] * kernel[..., np.newaxis]
        counts[y:y+inpt_size, x:x+inpt_size] += kernel

    return output / counts[..., np.newaxis]


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

def do_args(arg_list, name):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument("--input", action="store", dest="input_fn", type=str, required=True, \
        help="Input file name (pair list CSV file with 'naip_path', 'lc_path', 'nlcd_path' columns)"
    )
    parser.add_argument("--output", action="store", dest="output_base", type=str, required=True, \
        help="Output directory to store predictions"
    )
    parser.add_argument("--model", action="store", dest="model_fn", type=str, required=True, \
        help="Path to Keras .h5 model file to use"
    )
    parser.add_argument("--compress", action="store_true", default=False, \
        help="Enable GDAL compression (requires gdal_translate)"
    )
    parser.add_argument("--grayscale_output", action="store_true", default=False, \
        help="Enable outputing grayscale probability maps for each class"
    )
    parser.add_argument("--gpu", action="store", dest="gpu", type=int, required=False, \
        help="GPU id to use",
    )
    parser.add_argument("--superres", action="store_true", dest="superres", default=False, \
        help="Is this a superres model",
    )

    return parser.parse_args(arg_list)

def main():
    program_name = "Model inference script"
    args = do_args(sys.argv[1:], program_name)

    input_fn = args.input_fn
    output_base = args.output_base
    model_fn = args.model_fn
    compress = args.compress
    grayscale_output = args.grayscale_output
    superres = args.superres

    print("Starting %s at %s" % (program_name, str(datetime.datetime.now())))
    start_time = float(time.time())

    try:
        input_df = pd.read_csv(input_fn)
        pair_list = input_df[["naip_path", "nlcd_path", "lc_path"]].values
    except Exception as e:
        print("Could not load the input file")
        print(e)
        return

    model = keras.models.load_model(model_fn, custom_objects={
        "jaccard_loss":keras.metrics.mean_squared_error,
        "loss":keras.metrics.mean_squared_error
    })

    if superres:
        model = keras.models.Model(input=model.inputs, outputs=[model.outputs[0]])
        model.compile("sgd","mse")
    
    output_shape = model.output_shape[1:]
    input_shape = model.input_shape[1:]
    model_input_size = input_shape[0]

    print(input_shape, output_shape)
    print(model.outputs)

    for i in range(len(pair_list)):
        tic = float(time.time())
        naip_fn = pair_list[i][0]
        nlcd_fn = pair_list[i][1]
        lc_fn   = pair_list[i][2]

        print("Running model on %s\t%d/%d" % (naip_fn, i+1, len(pair_list)))

        naip_fid = rasterio.open(naip_fn, "r")
        naip_profile = naip_fid.meta.copy()
        naip_tile = naip_fid.read().astype(np.float32) / 255.0
        naip_tile = np.rollaxis(naip_tile, 0, 3)
        naip_fid.close()

        output = run_model_on_tile(model, naip_tile, model_input_size, output_shape[2], 32)
        #output[:,:,4] += output[:,:,5]
        #output[:,:,4] += output[:,:,6]
        output = output[:,:,:5]
        #----------------------------------------------------------------
        # Write out each softmax prediction to a separate file
        #----------------------------------------------------------------
        if grayscale_output:
            output_fn = os.path.basename(naip_fn)[:-4] + "_prob.tif"
            output_compressed_fn = output_fn[:-4] + "_compressed.tif"
            current_profile = naip_profile.copy()
            current_profile['driver'] = 'GTiff'
            current_profile['dtype'] = 'uint8'
            current_profile['count'] = 5

            bins = np.arange(256)
            bins = bins / 255.0
            output = np.digitize(output, bins=bins, right=True).astype(np.uint8)

            f = rasterio.open(os.path.join(output_base, output_fn), 'w', **current_profile)
            f.write(output[:,:,0], 1)
            f.write(output[:,:,1], 2)
            f.write(output[:,:,2], 3)
            f.write(output[:,:,3], 4)
            f.write(output[:,:,4], 5)
            f.close()

            # Run gdal_translate to compress the file we just wrote
            if compress:
                command = [
                    "gdal_translate",
                    "-of", "GTiff",
                    "-co", "INTERLEAVE=BAND",
                    "-co", "COMPRESS=DEFLATE",
                    "-co", "PREDICTOR=1",
                    "-co", "ZLEVEL=7",
                    "-co", "TILED=YES",
                    "-co", "BIGTIFF=YES",
                    "-co", "NUM_THREADS=ALL_CPUS",
                    "--config", "GDAL_CACHEMAX", "2048",
                    os.path.join(output_base, output_fn), os.path.join(output_base, output_compressed_fn)
                ]
                subprocess.call(command)


        #----------------------------------------------------------------
        # Write out the class predictions
        #----------------------------------------------------------------
        output_classes = np.argmax(output, axis=2).astype(np.uint8)
        output_class_fn = os.path.basename(naip_fn)[:-4] + "_class.tif"
        output_class_compressed_fn = output_class_fn[:-4] + "_compressed.tif"

        current_profile = naip_profile.copy()
        current_profile['driver'] = 'GTiff'
        current_profile['dtype'] = 'uint8'
        current_profile['count'] = 1
        f = rasterio.open(os.path.join(output_base, output_class_fn), 'w', **current_profile)
        f.write(output_classes, 1)
        f.close()

        # Run gdal_translate to compress the file we just wrote
        if compress:
            command = [
                "gdal_translate",
                "-of", "GTiff",
                "-co", "INTERLEAVE=BAND",
                "-co", "COMPRESS=DEFLATE",
                "-co", "PREDICTOR=1",
                "-co", "ZLEVEL=7",
                "-co", "TILED=YES",
                "-co", "BIGTIFF=YES",
                "-co", "NUM_THREADS=ALL_CPUS",
                "--config", "GDAL_CACHEMAX", "2048",
                os.path.join(output_base, output_class_fn), os.path.join(output_base, output_class_compressed_fn)
            ]
            subprocess.call(command)

        print("Finished iteration in %0.4f seconds" % (time.time() - tic))

    print("Finished %s in %0.4f seconds" % (program_name, time.time() - start_time))

if __name__ == "__main__":
    main()
