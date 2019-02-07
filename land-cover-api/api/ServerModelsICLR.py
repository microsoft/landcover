import sys, os, time

import numpy as np
import cntk

import fiona
import fiona.transform
import shapely
import shapely.geometry
import rasterio
import rasterio.mask

cnn_model_file = "data/cnn_60.model" #/mnt/blobfuse/train-output/ForKolyaPaper-train_Chesapeake2014_region_patch_LC4_sp2_NLCD-0.2-40/cnn_60.model

def load_cnn_model(fn, gpu_id=0):
    cntk.try_set_default_device(cntk.gpu(gpu_id))
    cntk.use_default_device()
    return cntk.load_model(fn)

def softmax(output):
    output_max = np.max(output, axis=3, keepdims=True)
    exps = np.exp(output-output_max)
    exp_sums = np.sum(exps, axis=3, keepdims=True)
    return exps/exp_sums

def run(naip, fn, extent, buffer):
    return run_cnn(naip)

def run_cnn(naip, batch_size=16):
    naip = naip/255.0
    #landsat = landsat/65536.0

    naip = np.swapaxes(naip, 0, 1)
    naip = np.swapaxes(naip, 0, 2)
    #landsat = np.swapaxes(landsat, 0, 1)
    #landsat = np.swapaxes(landsat, 0, 2)
    #blg = np.swapaxes(blg, 0, 1)
    #blg = np.swapaxes(blg, 0, 2)
    x1, x2, x3 = naip.shape
    #y1, y2, y3 = landsat.shape
    #z1, z2, z3 = blg.shape
    #s2 = min(x2, y2, z2)
    #s3 = min(x3, y3, z3)
    s2 = x2
    s3 = x3

    #tile = np.concatenate((naip[:, :s2, :s3], landsat[:, :s2, :s3], blg[:, :s2, :s3]),
    #        axis=0).astype(np.float32)
    #tile = tile[:, 21:-21, 21:-21]

    tile = naip[:, :s2, :s3]

    height = tile.shape[1]
    width = tile.shape[2]
    (_, model_width, model_height) = model.arguments[0].shape
    focus_rad = 32

    stride_x = model_width - focus_rad*2
    stride_y = model_height - focus_rad*2

    output = np.zeros((height, width, 5), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32) + 0.0001
    kernel = np.ones((model_height, model_width), dtype=np.float32) * 0.01
    kernel[10:-10, 10:-10] = 1
    kernel[focus_rad:focus_rad+stride_y, focus_rad:focus_rad+stride_x] = 5

    batch = []
    batch_indices = []
    batch_count = 0

    for y_index in (list(range(0, height - model_height, stride_y)) + [height - model_height,]):
        for x_index in (list(range(0, width - model_width, stride_x)) + [width - model_width,]):
            batch_indices.append((y_index, x_index))
            batch.append(tile[:, y_index:y_index+model_height, x_index:x_index+model_width])
            batch_count+=1

            if batch_count == batch_size:
                # run batch
                model_output = model.eval(np.array(batch))
                model_output = np.swapaxes(model_output,1,3)
                model_output = np.swapaxes(model_output,1,2)
                model_output = softmax(model_output)

                for i, (y, x) in enumerate(batch_indices):
                    output[y:y+model_height, x:x+model_width] += model_output[i] * kernel[..., np.newaxis]
                    counts[y:y+model_height, x:x+model_width] += kernel

                # reset batch
                batch = []
                batch_indices = []
                batch_count = 0

    if batch_count > 0:
        model_output = model.eval(np.array(batch))
        model_output = np.swapaxes(model_output,1,3)
        model_output = np.swapaxes(model_output,1,2)
        model_output = softmax(model_output)
        for i, (y, x) in enumerate(batch_indices):
            output[y:y+model_height, x:x+model_width] += model_output[i] * kernel[..., np.newaxis]
            counts[y:y+model_height, x:x+model_width] += kernel

    out = output/counts[..., np.newaxis]
    #out[:, :, 4] += out[:, :, 5] + out[:, :, 6]
    #out[:, :, 5:] = 0

    return out[..., 1:], cnn_model_file

model = load_cnn_model(cnn_model_file, 0)
print("CNN loaded")

