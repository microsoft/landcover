import argparse
import numpy as np
import rasterio
import json
from pathlib import Path
import os

import pdb

import torch
import torch.nn as nn
from training.pytorch.models.unet import Unet
from training.pytorch.utils.eval_segm import mean_IoU, pixel_accuracy
from training.pytorch.utils.data.load_tile import load_tile
from training.pytorch.data_loader import DataGenerator
from torch.utils import data
from torch.autograd import Variable
from einops import rearrange

from web_tool.ServerModelsNIPSGroupNorm import GroupParams

parser = argparse.ArgumentParser()


if __name__ == '__main__':
    parser.add_argument('--config_file', type=str, default="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/params.json", help="json file containing the configuration")

    parser.add_argument('--model_file', type=str,
                    help="Checkpoint saved model",
                    default="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar")
    parser.add_argument('--test_tile_fn', type=str, help="Filename with tile file names in npy format", default="training/data/finetuning/test1_test_tiles.txt")  #   training/data/finetuning/val1_test_patches.txt
    parser.add_argument('--tile_type', type=str, help="Filename with tile file names in npy format", default="test")
    parser.add_argument('--area', type=str, help="Name of area being tested in: test1, test2, test3, test4, or val1", default="test1")


    args = parser.parse_args()



def predict_entire_image_unet_fine(model, x):
    # x: (height, width, channel)
    if torch.cuda.is_available():
        model.cuda()
    norm_image = x
    norm_image = rearrange(norm_image, 'height width channel -> channel height width')
    c, h, w = norm_image.shape
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        margin = model.border_margin_px
    except:
        margin = 0
    
    out = np.zeros((5, h, w))
    # (channel, height, width)

    patch_dimension = 892

    stride = patch_dimension - 2 * margin

    max_x = 0
    max_y = 0
    
    for x in range(0, w, stride):
        for y in range(0, h, stride):
            patch = norm_image[:4, y:y+patch_dimension, x:x+patch_dimension]
            c, h1, w1 = patch.shape
            if not (h1 == patch_dimension and w1 == patch_dimension):
            #    pdb.set_trace()
                continue
            patch_tensor = torch.from_numpy(patch).float().to(device)
            y_pred1 = model.forward(patch_tensor.unsqueeze(0))
            _, c_y, h_y, w_y = y_pred1.shape
            if not (h_y == stride and w_y == stride):
            #    pdb.set_trace()
                continue
            y_hat1 = (Variable(y_pred1).data).cpu().numpy()
            y_hat1 = y_hat1.squeeze(0)
            out[:, y + margin:y + patch_dimension - margin, x + margin : x + patch_dimension - margin] = y_hat1 # [:, y + margin:y + patch_dimension - margin, x + margin : x + patch_dimension - margin]
            max_x = x + patch_dimension - margin
            max_y = y + patch_dimension - margin
    
    for i in range(0,h, stride):
        patch = norm_image[:4, i:i+patch_dimension, w-patch_dimension:w]
        c, h1, w1 = patch.shape
        if not (h1 == patch_dimension and w1 == patch_dimension):
            #    pdb.set_trace()
            continue
        patch_tensor = torch.from_numpy(patch).float().to(device)
        y_pred1 = model.forward(patch_tensor.unsqueeze(0))
        _, c_y, h_y, w_y = y_pred1.shape
        if not (h_y == stride):
            #    pdb.set_trace()
            continue
        y_hat1 = (Variable(y_pred1).data).cpu().numpy()
        y_hat1 = y_hat1.squeeze(0)
        out[:, i + margin:i + patch_dimension - margin, w-patch_dimension + margin : w - margin] = y_hat1 # [:, y + margin:y + patch_dimension - margin, x + margin : x + patch_dimension - margin]

    for j in range(0,w, stride):
        patch = norm_image[:4, h-patch_dimension:h, j:j+patch_dimension]
        c, h1, w1 = patch.shape
        if not (h1 == patch_dimension and w1 == patch_dimension):
            #    pdb.set_trace()
            continue
        patch_tensor = torch.from_numpy(patch).float().to(device)
        y_pred1 = model.forward(patch_tensor.unsqueeze(0))
        _, c_y, h_y, w_y = y_pred1.shape
        if not (h_y == stride):
            #    pdb.set_trace()
            continue
        y_hat1 = (Variable(y_pred1).data).cpu().numpy()
        y_hat1 = y_hat1.squeeze(0)
        out[:, h-patch_dimension + margin:h - margin, j + margin : j + patch_dimension - margin] = y_hat1     
    #pred = np.rollaxis(out, 0, 3)   # (w, h, c)
    #pred = np.moveaxis(pred, 0, 1)  # (h, w, c)
    patch = norm_image[:4, h-patch_dimension:h, w-patch_dimension:w]
    c, h1, w1 = patch.shape
    patch_tensor = torch.from_numpy(patch).float().to(device)
    y_pred1 = model.forward(patch_tensor.unsqueeze(0))
    y_hat1 = (Variable(y_pred1).data).cpu().numpy()
    y_hat1 = y_hat1.squeeze(0)
    out[:, h-patch_dimension + margin:h - margin, w-patch_dimension + margin:w - margin] = y_hat1
    pred = rearrange(out, 'channel height width -> height width channel')
    return pred


def run_model_on_tile(model, naip_tile, output_file_path=None, batch_size=32):
    # (height, width, channel)
    y_hat = predict_entire_image_unet_fine(model, naip_tile)
    # (h, w, c)
    if output_file_path:
        np.save(output_file_path, y_hat)
    out =np.argmax(y_hat, axis=-1)
    pdb.set_trace()
    import scipy.misc
    scipy.misc.imsave('pred.jpg', image_array)
    return np.argmax(y_hat, axis=-1)
    # (h, w)



def run(model, naip_data, output_file_path=None):
    # apply padding to the output_features
    # naip_data: (batch, channel, height, width)
    x = np.squeeze(naip_data, 0)
    # (channel, height, width)
    x = np.swapaxes(x, 0, 2)
    # (width, height, channel)
    x = np.swapaxes(x, 0, 1)
    # (height, width, channel)

    x = x[:, :, :4]
    naip_data = x
    # (height, width, channel)
    output = run_model_on_tile(model, naip_data, output_file_path=output_file_path)
    # (height, width)
    return output


def load_model(path_2_saved_model, model_opts, outer_class=None):
    checkpoint = torch.load(path_2_saved_model)
    model = Unet(model_opts)
    if outer_class:
        model = outer_class(model)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def main(model_file, config_file, area, tile_list_file_name, tile_type):
    params = json.load(open(config_file, "r"))
    is_group_params = ('group_params' in model_file)
    
    model = load_model(model_file, params['model_opts'], outer_class=(GroupParams if is_group_params else None))
    
    f = open(tile_list_file_name, "r")
    test_tiles_files = f.read().strip().split("\n")
    f.close()
    
    # batch_size = params["loader_opts"]["batch_size"]
    # num_channels = params["loader_opts"]["num_channels"]
    # model_opts = params["model_opts"]

    running_mean_IoU = 0
    running_pixel_accuracy = 0

    for i, test_tile in enumerate(test_tiles_files):
        tile = load_tile(test_tile.replace('.mrf', '.npy'))
        # (batch, channel, height, width)
        prediction_file_path = model_file.replace('.tar', '_predictions/%s_%d_%s.npy' % (args.tile_type, i, Path(test_tile).name))
        os.makedirs(str(Path(prediction_file_path).parent), exist_ok=True)
        result = run(model, tile, prediction_file_path)
        # (height, width)

        y_train_hr = tile[0, 4, :, :]
        height, width = y_train_hr.shape
        
        margin = model.border_margin_px
        tile_mean_IoU = mean_IoU(result[margin:height-margin, margin:width-margin], y_train_hr[margin:height-margin, margin:width-margin], ignored_classes={0})
        tile_pixel_accuracy = pixel_accuracy(result[margin:height-margin, margin:width-margin], y_train_hr[margin:height-margin, margin:width-margin], ignored_classes={0})


        print('%s, %s, %s, %d, %f, %f, %s, %s,' % (Path(model_file).name, area, tile_type, i, tile_mean_IoU, tile_pixel_accuracy, test_tile, prediction_file_path))
        
    

if __name__ == '__main__':
    main(args.model_file, args.config_file, args.area, args.test_tile_fn, args.tile_type)

