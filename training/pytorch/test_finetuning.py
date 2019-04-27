import argparse
import numpy as np
import rasterio
import json

import pdb

import torch
import torch.nn as nn
from training.pytorch.models.unet import Unet
from training.pytorch.utils.eval_segm import mean_IoU
from training.pytorch.data_loader import DataGenerator
from torch.utils import data
from torch.autograd import Variable
from einops import rearrange

parser = argparse.ArgumentParser()

parser.add_argument('--config_file', type=str, default="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/params.json", help="json file containing the configuration")
parser.add_argument('--model_file', type=str,
                    help="Checkpoint saved model",
                    default="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar")
parser.add_argument('--test_tile_fn', type=str, help="Filename with tile file names in npy format", default="training/data/finetuning/test1.txt")  #   training/data/finetuning/val1_test_patches.txt

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
    
    out = np.zeros((5, h - 2 * margin, w - 2 * margin))
    # (channel, height, width)

    patch_dimension = 892

    stride = patch_dimension - 2 * margin
    
    for x in range(0, w, stride):
        for y in range(0, h, stride):
            patch = norm_image[:4, y:y+patch_dimension, x:x+patch_dimension]
            c, h1, w1 = patch.shape
            if not (h1 == patch_dimension and w1 == patch_dimension):
                continue
            patch_tensor = torch.from_numpy(patch).float().to(device)
            y_pred1 = model.forward(patch_tensor.unsqueeze(0))
            y_hat1 = (Variable(y_pred1).data).cpu().numpy()
            
            out[:, y + margin:y + patch_dimension - margin, x + margin : x + patch_dimension - margin] = y_hat1
    #pred = np.rollaxis(out, 0, 3)   # (w, h, c)
    #pred = np.moveaxis(pred, 0, 1)  # (h, w, c)
    pred = rearrange(out, 'channel height width -> height width channel')
    return pred


def run_model_on_tile(model, naip_tile, batch_size=32):
    # (height, width, channel)
    y_hat = predict_entire_image_unet_fine(model, naip_tile)
    # (h, w, c)
    return np.argmax(y_hat, axis=-1)
    # (h, w)



def run(model, naip_data):
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
    output = run_model_on_tile(model, naip_data)
    # (height, width)
    return output


def load_model(path_2_saved_model, model_opts):
    checkpoint = torch.load(path_2_saved_model)
    model = Unet(model_opts)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def main(model_file, config_file):
    params = json.load(open(config_file, "r"))    
    model = load_model(model_file, params['model_opts'])

    f = open(args.test_tile_fn, "r")
    test_tiles_files = f.read().strip().split("\n")
    f.close()
    
    # batch_size = params["loader_opts"]["batch_size"]
    # num_channels = params["loader_opts"]["num_channels"]
    # model_opts = params["model_opts"]

    running_mean_IoU = 0

    for test_tile in test_tiles_files:
        tile = np.load(test_tile.replace('.mrf', '.npy'))
        # (batch, channel, height, width)
        print(tile.shape)
        result = run(model, tile)
        # (height, width)
        
        y_train_hr = tile[0, :, :, 4]
        height, width = y_train_hr.shape
        # (height, width)
        
        # Collapse larger class space down to 4+1 primary classes {unknown, water, forest, field, built}
        y_train_hr[y_train_hr == 15] = 0
        y_train_hr[y_train_hr == 5] = 4
        y_train_hr[y_train_hr == 6] = 4
        
        running_mean_IoU += mean_IoU(y_train_hr[model.margin:height-model.margin, model.margin:width-model.margin], result, ignored_classes={0})

    running_mean_IoU /= len(test_tiles_files)
    
    print('%s\t%f' % (model_file, running_mean_IoU))


if __name__ == '__main__':
    main(args.model_file, args.config_file)
