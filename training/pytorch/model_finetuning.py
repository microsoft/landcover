import argparse
import numpy as np
from pprint import pprint
from attr import attrs, attrib
from einops import rearrange
import pdb
from datetime import datetime, timedelta
from pathlib import Path
from itertools import product
import csv
import math
import copy
import json
import os
import random
import heapq

import torch
import torch.nn as nn
from torch.utils import data
from torch.optim import lr_scheduler

from training.pytorch.models.unet import Unet
from training.pytorch.models.fusionnet import Fusionnet
from training.pytorch.utils.save_visualize import save_visualize
from training.pytorch.utils.eval_segm import mean_IoU, pixel_accuracy
from training.pytorch.utils.experiments_utils import improve_reproducibility
from training.pytorch.utils.filesystem import ensure_dir
from training.pytorch.losses import (multiclass_ce, multiclass_dice_loss, multiclass_jaccard_loss, multiclass_tversky_loss, multiclass_ce_points)
from training.pytorch.data_loader import DataGenerator
from training.pytorch.utils.data.tile_to_npy import sample
from training.pytorch.utils.data.load_tile import load_tile, labels, features
from training.pytorch.test_finetuning import predict_entire_image_unet_fine

from web_tool.ServerModelsNIPSGroupNorm import softmax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/params.json", help="json file containing the configuration")

    parser.add_argument('--model_file', type=str,
                   help="Checkpoint saved model",
                   default="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar")

    #parser.add_argument('--data_path', type=str, help="Path to data", default="/mnt/blobfuse/cnn-minibatches/summer_2019/active_learning_splits/")
    # parser.add_argument('--data_sub_dirs', type=str, nargs='+', help="Sub-directories of `data_path` to get data from", default=['val1',]) # 'test1', 'test2', 'test3', 'test4'])

    parser.add_argument('--run_validation', action="store_true", help="Whether to run validation")

    # parser.add_argument('--validation_patches_fn', type=str, help="Filename with list of training patch files", default="training/data/finetuning/val2_train_patches_5.txt")
    # parser.add_argument('--training_patches_fn', type=str, help="Filename with list of training patch files", default="training/data/finetuning/val2_train_patches_5.txt")

    parser.add_argument('--area', type=str, help="Name of area being tested in: test1, test2, test3, test4, or val1", default="test1")
    parser.add_argument('--train_tiles_list_file_name', type=str, help="Filename with list of training tile files", default="training/data/finetuning/test1_train_tiles.txt")
    parser.add_argument('--test_tiles_list_file_name', type=str, help="Filename with list of training tile files", default="training/data/finetuning/test1_test_tiles.txt")
    parser.add_argument('--log_fn', type=str, help="Where to store training results", default="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/val/val2/10_patches/finetune_results.csv")
    parser.add_argument('--model_output_directory', type=str, help='Where to store fine-tuned model', default='/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/val/val2_fix/')
    parser.add_argument('--random_seed', type=int, help="Random seed for reproducibility", default=0)
    parser.add_argument('--active_learning_strategy', type=str, help="Which point selection strategy to use for active learning query method", default='random')
    parser.add_argument('--active_learning_batch_size', type=int, help="How many points to select for each active learning query. If -1 (default), use progressive steps: [10, 30, 60, 100, 200, 600, 1000]", default=-1)
    
    args = parser.parse_args()

class GroupParams(nn.Module):

    def __init__(self, model):
        super(GroupParams, self).__init__()
        self.gammas = nn.Parameter(torch.ones((1, 32, 1, 1)))
        self.betas = nn.Parameter(torch.zeros((1, 32, 1, 1)))
        self.model = model
        try:
            self.border_margin_px = model.border_margin_px
        except:
            self.border_margin_px = 0

    def forward(self, x):
        x, conv1_out, conv1_dim = self.model.down_1(x)

        x, conv2_out, conv2_dim = self.model.down_2(x)

        x, conv3_out, conv3_dim = self.model.down_3(x)
        x, conv4_out, conv4_dim = self.model.down_4(x)

        # Bottleneck
        x = self.model.conv5_block(x)

        # up layers
        x = self.model.up_1(x, conv4_out, conv4_dim)
        x = self.model.up_2(x, conv3_out, conv3_dim)
        x = self.model.up_3(x, conv2_out, conv2_dim)
        x = self.model.up_4(x, conv1_out, conv1_dim)
        x = x * self.gammas + self.betas

        return self.model.conv_final(x)


@attrs
class FineTuneResult(object):
    best_mean_IoU = attrib(type=float)
    train_duration = attrib(type=timedelta)
    

def set_dropout(layers, dropout_state, original_params):
    for id in dropout_state:
        p = layers[id]
        for f in range(p.shape[1]):
            p[:,f] = original_params[id][:,f] * dropout_state[id][f]# / (np.average(dropout_state[id]))
            # can in one line

def clone_state(dropout_state):
    return { i : np.copy(dropout_state[i]) for i in dropout_state }

def train_model_dropout(model, criterion, dataloaders, hyper_parameters, log_writer, num_epochs=20, superres=False, masking=False):
    global results_writer, results_file

    dropout_rate = hyper_parameters['dropout_rate']

    last_k_layers = hyper_parameters['last_k_layers']
    
    # mask_id indices (points per patch): [1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80, 100]
    mask_id = hyper_parameters['mask_id']
    
    since = datetime.now()


    original_params = {}
    c_set = 0.

    dropout_state = {}
    
    dropout_layers = list(model.parameters())

    for i,p in list(enumerate(model.parameters()))[-1::-1]:
        if len(p.shape)==4 and p.shape[0] > 1 and i < 48:
            c_set += 1
            original_params[i] = p.clone()
            dropout_state[i] = np.ones((p.shape[1],))
        if c_set == last_k_layers: break
        
    best_dropout_state = clone_state(dropout_state)
    best_loss = 1000000.


    best_model_wts = copy.deepcopy(model.state_dict())
    best_mean_IoU = 0.0
    best_epoch = -1
    duration_til_best_epoch = since - since
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Each epoch can have a training and validation phase
    phases = dataloaders.keys()
    for phase in phases:
        if phase not in ['train', 'val']:
            print('Warning: epoch phase "%s" not valid. Valid options: ["train", "val"]. Data provided in this phase may be ignored.' % phase)

    for epoch in range(-1, num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)
        
        statistics = {
            'mean_IoU': -1,
            'loss': -1,
            'accuracy': -1
        }

        epoch_statistics = {
            phase: copy.deepcopy(statistics)
            for phase in phases
        }
        # print(epoch_statistics)

        # sample new state:
        if epoch > 0:
            for id in dropout_state:
                new_state = np.random.binomial(1, 1 - dropout_rate * np.random.ranf(), dropout_state[id].shape)
                mask = np.random.binomial(1, 1 - dropout_rate, dropout_state[id].shape)
                dropout_state[id] = best_dropout_state[id] * mask + new_state * (1. - mask)

            set_dropout(dropout_layers, dropout_state, original_params)

        hyper_parameters['epoch'] = epoch

        for phase in phases:
            if phase == 'train':
                pass
                #scheduler.step()
                #model.train()  # Set model to training mode
            else:  # phase == 'val'
                if 'val' in dataloaders:
                    model.eval()   # Set model to evaluate mode
                else:
                    continue
            epoch_statistics[phase]['loss'] = 0.0
            epoch_statistics[phase]['mean_IoU'] = 0.0
            epoch_statistics[phase]['accuracy'] = 0.0
            
            running_loss = 0.0
            meanIoU = 0.0
            accuracy = 0.0
            
            n_iter = 0

            # Iterate over data.
            for entry in dataloaders[phase]:
                if superres:
                    if masking:
                        inputs, labels, nlcd, masks = entry
                    else:
                        inputs, labels, nlcd = entry
                    # TODO: use nlcd for superres training, below
                else:
                    if masking:
                        inputs, labels, masks = entry
                    else:
                        inputs, labels = entry

                inputs = inputs[:, :, 2:240 - 2, 2:240 - 2]
                labels = labels[:, :, 94:240 - 94, 94:240 - 94]
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                if masking and phase == 'train':
                    masks = masks.float()
                    masks = masks.to(device)
                    masks = rearrange(masks, 'batch unknown masks height width -> batch (unknown masks) height width')
                    mask = masks[:, mask_id : mask_id + 1, 94:240 - 94, 94:240 - 94].to(device)
                    labels = labels * mask


                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train' and epoch > -1):
                    outputs = model.forward(inputs)
                    ground_truth = torch.squeeze(labels,1).long()
                    # print(outputs.shape)
                    # print(ground_truth.shape)
                    path = str(Path(args.model_output_directory) / ("epoch_" + str(epoch) + "_" + phase))
                    ensure_dir(path)
                    # print('Save to path: %s' % path)
                    # save_visualize(inputs, outputs, ground_truth, path)

                    # pdb.set_trace() # print(outputs.shape, ground_truth.shape)
                    loss = criterion(ground_truth, outputs)


                # Store ground truth
                y_hr = np.squeeze(labels.cpu().numpy(), axis=1)
                # TODO: I think we need the below... causes error though:
                #if phase == 'train':
                    #y_hr = y_hr * mask.cpu().detach().numpy()
                
                # Store current outputs
                batch_size, _, _ = y_hr.shape
                # TODO: do we need this check below?
                if phase == 'train':
                    y_hat = outputs.cpu().detach().numpy() # * mask.cpu().detach().numpy()
                else:
                    y_hat = outputs.cpu().numpy()
                y_hat = np.argmax(y_hat, axis=1)

                
                # statistics
                n_iter += 1

                # 1) Loss
                epoch_statistics[phase]['loss'] += loss.item()
                
                # 2) mean_IoU
                batch_meanIoU = 0
                for j in range(batch_size):
                    batch_meanIoU += mean_IoU(y_hat[j], y_hr[j], ignored_classes={0})
                batch_meanIoU /= batch_size
                epoch_statistics[phase]['mean_IoU'] += batch_meanIoU
                
                # 3) accuracy
                batch_accuracy = 0
                for j in range(batch_size):
                    batch_accuracy += pixel_accuracy(y_hat[j], y_hr[j], ignored_classes={0})
                batch_accuracy /= batch_size
                epoch_statistics[phase]['accuracy'] += batch_accuracy
                
            # Normalize statistics per training iteration in epoch
            for key in epoch_statistics[phase]:
                epoch_statistics[phase][key] /= n_iter  # divide by how many batches were processed in this epoch

            # print('number of batches in epoch', len(dataloaders[phase]))
            # print('n_iter', n_iter)

            #print(phase, epoch, epoch_statistics[phase])

        if epoch_statistics['train']['loss'] < best_loss:
            best_dropout_state = clone_state(dropout_state)
            best_loss = epoch_statistics['train']['loss']
        
        dropout_state = clone_state(best_dropout_state)
                
        result_row = {
            'run_id': hyper_parameters['run_id'],
            'hyper_parameters': hyper_parameters,
            'epoch': epoch,
            'train_loss': epoch_statistics['train']['loss'],
            'train_accuracy': epoch_statistics['train']['accuracy'],
            'train_mean_IoU': epoch_statistics['train']['mean_IoU'],
            'val_loss': epoch_statistics['val']['loss'] if 'val' in epoch_statistics else None,
            'val_accuracy': epoch_statistics['val']['accuracy'] if 'val' in epoch_statistics else None,
            'val_mean_IoU': epoch_statistics['val']['mean_IoU'] if 'val' in epoch_statistics else None,
            'total_time': datetime.now() - since
        }
        print(result_row)
        results_writer.writerow(result_row)
        results_file.flush()

        model_file_name_suffix = hyper_parameters_str_dropout(hyper_parameters)
        
        finetuned_fn = str(Path(args.model_output_directory) / ("finetuned_unet_gn.pth_%s.tar" % model_file_name_suffix))

        torch.save(model.state_dict(), finetuned_fn)
        
        # deep copy the model
            #if phase == 'val' and epoch_mean_IoU > best_mean_IoU:
            #    best_mean_IoU = epoch_mean_IoU
            #    best_model_wts = copy.deepcopy(model.state_dict())
            #    best_epoch = epoch
            #    duration_til_best_epoch = datetime.now() - since
        # print()
        
    # for area, tile_file_names in zip(args.test_areas, args.test_tile_list_file_names):
    # test_model(finetuned_fn, args.config_file, args.area, args.train_tile_list_file_names, 'train')
    # test_model(finetuned_fn, args.config_file, args.area, args.test_tile_list_file_names, 'test')

    set_dropout(dropout_layers, dropout_state, original_params) # and this is the best model

    duration = datetime.now() - since
    seconds_elapsed = duration.total_seconds()
    
    ## print('Training complete in {:.0f}m {:.0f}s'.format(
    #    seconds_elapsed // 60, seconds_elapsed % 60))
    ## print('Best val IoU: {:4f}'.format(best_mean_IoU))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, FineTuneResult(best_mean_IoU=best_mean_IoU, train_duration=duration)



def OLD(path_2_saved_model, loss, gen_loaders, params, hyper_parameters, log_writer, n_epochs=25):

    best_dropout_state = clone_state(dropout_state)
    best_loss = 0#1000000.
    
    phases = ['train']#, 'val']
    

    for epoch in range(n_epochs):
        #print('EPOCH {}: '.format(epoch))
        #print('current dropout strength: ' + ' '.join([str((i, 1.-np.average(dropout_state[i]))) for i in sorted(dropout_state)]))
        
        statistics = {
            'mean_IoU': -1,
            'loss': -1,
            'accuracy': -1
        }
        
        epoch_statistics = {
            phase: copy.deepcopy(statistics)
            for phase in phases
        }
        #print(epoch_statistics)


        # sample new state:
        if epoch > 0:
        
            for id in dropout_state:
                new_state = np.random.binomial(1, 1 - dropout_rate * np.random.ranf(), dropout_state[id].shape)
                mask = np.random.binomial(1, 1 - dropout_rate, dropout_state[id].shape)
                dropout_state[id] = best_dropout_state[id] * mask + new_state * (1. - mask)

            set_dropout(dropout_layers, dropout_state, original_params)

        
        for phase in phases:

            epoch_statistics[phase]['loss'] = 0.0
            epoch_statistics[phase]['mean_IoU'] = 0.0
            epoch_statistics[phase]['accuracy'] = 0.0
            
            running_loss = 0.0
            meanIoU = 0.0
            accuracy = 0.0

            model.eval()
        

            n_iter = 0
            for entry in gen_loaders[phase]:
                inputs, labels, masks = entry

                inputs = inputs[:, :, 2:240 - 2, 2:240 - 2]
                labels = labels[:, :, 94:240 - 94, 94:240 - 94]
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == 'train':
                    masks = masks.float()
                    masks = masks.to(device)
                    masks = rearrange(masks, 'batch unknown masks height width -> batch (unknown masks) height width')
                    mask = masks[:, mask_id : mask_id + 1, 94:240 - 94, 94:240 - 94].to(device)
                    labels = labels * mask

                outputs = model.forward(inputs)
                ground_truth = torch.squeeze(labels,1).long()

                this_loss = loss(ground_truth, outputs)

                n_iter += 1

                y_hr = np.squeeze(labels.cpu().numpy(), axis=1)
                
                batch_size, _, _ = y_hr.shape
                
                if phase == 'train':
                    y_hat = outputs.cpu().detach().numpy() * mask.cpu().detach().numpy()
                else:
                    y_hat = outputs.cpu().detach().numpy()
                y_hat = np.argmax(y_hat, axis=1)

                epoch_statistics[phase]['loss'] += this_loss.item()

                #print(this_loss.item())
                
                batch_meanIoU = 0
                for j in range(batch_size):
                    batch_meanIoU += mean_IoU(y_hat[j], y_hr[j], ignored_classes={0})
                batch_meanIoU /= batch_size
                epoch_statistics[phase]['mean_IoU'] += batch_meanIoU
                
                batch_accuracy = 0
                for j in range(batch_size):
                    batch_accuracy += pixel_accuracy(y_hat[j], y_hr[j], ignored_classes={0})
                batch_accuracy /= batch_size
                epoch_statistics[phase]['accuracy'] += batch_accuracy
            
            # Normalize statistics per training iteration in epoch
            for key in epoch_statistics[phase]:
                epoch_statistics[phase][key] /= n_iter  # divide by how many batches were processed in this epoch

            #print('number of batches in epoch', len(gen_loaders[phase]))
            #print('n_iter', n_iter)

            print(phase, epoch, epoch_statistics[phase])

            ## retrain

        #print(epoch_statistics['train']['loss'], best_loss)
        if epoch_statistics['train']['accuracy'] > best_loss:
            best_dropout_state = clone_state(dropout_state)
            #best_loss = epoch_statistics['train']['loss']
            best_loss = epoch_statistics['train']['accuracy']
        
        dropout_state = clone_state(best_dropout_state)
        
            
        # result_row = {
        #     'run_id': hyper_parameters['run_id'],
        #     'hyper_parameters': hyper_parameters,
        #     'epoch': epoch,
        #     'train_loss': epoch_statistics['train']['loss'],
        #     'train_accuracy': epoch_statistics['train']['accuracy'],
        #     'train_mean_IoU': epoch_statistics['train']['mean_IoU'],
        #     'val_loss': epoch_statistics['val']['loss'],
        #     'val_accuracy': epoch_statistics['val']['accuracy'],
        #     'val_mean_IoU': epoch_statistics['val']['mean_IoU'],
        #     'total_time': datetime.now() - since
        # }
        # print(result_row)
        # results_writer.writerow(result_row)
        # results_file.flush()

    set_dropout(dropout_layers, dropout_state, original_params) # and this is the best model

    finetuned_fn = str(Path(args.model_output_directory) / ("finetuned_unet_gn.dropout_final.tar"))
    torch.save(unet.state_dict(), finetuned_fn)

    duration = datetime.now() - since
    seconds_elapsed = duration.total_seconds()
    
    return unet, FineTuneResult(best_mean_IoU=-1, train_duration=duration)

    
def finetune_group_params(path_2_saved_model, loss, gen_loaders, params, params_train, hyper_parameters, log_writer, n_epochs=25):
    learning_rate = hyper_parameters['learning_rate']
    optimizer_method = hyper_parameters['optimizer_method']
    lr_schedule_step_size = hyper_parameters['lr_schedule_step_size']
    if 'epochs' in hyper_parameters:
        n_epochs = hyper_parameters['epochs']

    def load_model():
        opts = params["model_opts"]
        unet = Unet(opts)
        checkpoint = torch.load(path_2_saved_model)
        unet.load_state_dict(checkpoint['model'])
        unet.eval()
        for param in unet.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        model_2_finetune = GroupParams(unet)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_2_finetune = model_2_finetune.to(device)
        loss_fun = loss().to(device)

        optimizer = torch.optim.SGD(model_2_finetune.parameters(), lr=learning_rate, momentum=0.9)
        if optimizer_method == torch.optim.Adam:
            optimizer = torch.optim.Adam(model_2_finetune.parameters(), lr=learning_rate, eps=1e-5)

        return model_2_finetune, loss_fun, optimizer

    model_2_finetune, loss_fun, optimizer = load_model()
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_schedule_step_size, gamma=0.1)

    model_2_finetune = active_learning(load_model, loss_fun, optimizer,
                                       exp_lr_scheduler, gen_loaders, params, params_train, hyper_parameters, log_writer, num_epochs=n_epochs)
    return model_2_finetune

def finetune_dropout(path_2_saved_model, loss, gen_loaders, params, params_train, hyper_parameters, log_writer, n_epochs=25):
    dropout_rate = hyper_parameters['dropout_rate']

    last_k_layers = hyper_parameters['last_k_layers']

    if 'epochs' in hyper_parameters:
        n_epochs = hyper_parameters['epochs']

    def load_model():
        opts = params["model_opts"]
        unet = Unet(opts)
        checkpoint = torch.load(path_2_saved_model)
        unet.load_state_dict(checkpoint['model'])
        unet.eval()
        
        for param in unet.parameters():
            param.requires_grad = False
    
        # Parameters of newly constructed modules have requires_grad=True by default
        model_2_finetune = unet
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_2_finetune = model_2_finetune.to(device)

        loss_fun = loss().to(device)

        
        return model_2_finetune, loss_fun

    model_2_finetune, loss_fun = load_model()
        
    
    model_2_finetune = active_learning_dropout(load_model, loss_fun, gen_loaders, params, params_train, hyper_parameters, log_writer, num_epochs=n_epochs)
    return model_2_finetune


def finetune_last_k_layers(path_2_saved_model, loss, gen_loaders, params, params_train, hyper_parameters, log_writer, n_epochs=25):
    learning_rate = hyper_parameters['learning_rate']
    optimizer_method = hyper_parameters['optimizer_method']
    lr_schedule_step_size = hyper_parameters['lr_schedule_step_size']
    last_k_layers = hyper_parameters['last_k_layers']
    if 'epochs' in hyper_parameters:
        n_epochs = hyper_parameters['epochs']

    def load_model():
        opts = params["model_opts"]
        unet = Unet(opts)
        checkpoint = torch.load(path_2_saved_model)
        unet.load_state_dict(checkpoint['model'])
        unet.eval()
        
        for layer in list(unet.children())[:-last_k_layers]:
            for param in layer.parameters():
                param.requires_grad = False
    
        # Parameters of newly constructed modules have requires_grad=True by default
        model_2_finetune = unet
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_2_finetune = model_2_finetune.to(device)

        loss_fun = loss().to(device)

        optimizer = torch.optim.SGD(model_2_finetune.parameters(), lr=learning_rate, momentum=0.9)
        if optimizer_method == torch.optim.Adam:
            optimizer = torch.optim.Adam(model_2_finetune.parameters(), lr=learning_rate, eps=1e-5)
        
        return model_2_finetune, loss_fun, optimizer

    model_2_finetune, loss_fun, optimizer = load_model()
        
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_schedule_step_size, gamma=0.1)

    model_2_finetune = active_learning(load_model, loss_fun, optimizer,
                                       exp_lr_scheduler, gen_loaders, params, params_train, hyper_parameters, log_writer, num_epochs=n_epochs)
    return model_2_finetune


def active_learning_step_size(step_num):
    if args.active_learning_batch_size > 0:
        return args.active_learning_batch_size
    else:
        step_sizes = [10, 30, 60, 100, 200, 600, 1000]
        # number of points will be 0, 10, 40, 100, 200, 400, 1000, 2000
        return step_sizes[step_num]
    
    #if num_points < 200:
    #    return 50
    #if num_points < 1000:
    #    return 100
    #if num_points < 10000:
    #    return 1000


def prediction_entropy(predictions):
    # predictions: (height, width, channels)
    predictions = softmax(predictions)
    return -(predictions * np.log(predictions)).sum(axis=-1)


def pixels_to_patches(train_tile, points):
    # train_tile: (batch, channels, height, width)
    # return one 240 x 240 patch per point
    _, num_channel, tile_height, tile_width = train_tile.shape
    patch_height = 240
    patch_width = 240
    # Patches dimensions will be (B, C, H, W)
    patches = []
    mask = np.zeros((patch_height, patch_width))
    mask[patch_height // 2][patch_width // 2] = 1
    #Fixme: Check for edge cases where points are close to the border. We might not want this zero padding
    for i, point in enumerate(points):
        row, col = point
        patch = copy.deepcopy(train_tile[0, :, row-patch_height//2:row+patch_height//2, col-patch_width//2:col+patch_width//2])
        # pdb.set_trace()
        patch[4] *= mask
        patch[5] *= mask
        # Only provide labels available at the point selected
        patches.append(patch)
    return patches


####################################################
####### Active Learning Query Strategies ###########
####################################################


def get_query_strategy(active_learning_strategy_str):
    query_strategy = None
    if active_learning_strategy_str == 'random':
        query_strategy = random_selection
    if active_learning_strategy_str == 'entropy':
        query_strategy = entropy_selection
    if active_learning_strategy_str == 'margin':
        query_strategy = margin_selection
    if active_learning_strategy_str == 'mistake_random':
        query_strategy = mistake_random_selection
    if active_learning_strategy_str == 'mistake_entropy':
        query_strategy = mistake_entropy_selection
    if active_learning_strategy_str == 'mistake_margin':
        query_strategy = mistake_margin_selection
    if active_learning_strategy_str == 'half_mistake':
        query_strategy = half_mistake_selection
    if query_strategy == None:
        query_strategy = random_selection
        print('No query_strategy specified: using random_selection')
    return query_strategy

def random_sample(points, num):
    # Return `num` elements from `points` at random; return the whole set if not enough
    try:
        return random.sample(points, num)
    except:
        return points


def entropy_selection(train_tile, predictions, possible_indices, num_new_patches):
    # predictions: (height, width, channels)
    entropy = prediction_entropy(predictions)
    # entropy: (height, width)

    highest_entropy_points = heapq.nlargest(num_new_patches,
                                            possible_indices,
                                            key=lambda index: entropy[index])

    return highest_entropy_points # new_train_patches


def margin_selection(train_tile, predictions, possible_indices, num_new_patches):
    # predictions: (height, width, channels)
    predictions = softmax(predictions) # logits to probabilities
    sorted_predictions = np.sort(predictions, axis=-1) # sort probabilities, smallest to largest
    margin = sorted_predictions[:, :, -1] - sorted_predictions[:, :, -2]  # (largest prob) - (2nd largest prob)
    
    lowest_margin_points = heapq.nsmallest(num_new_patches,
                                           possible_indices,
                                           key=lambda index: margin[index])

    return lowest_margin_points


def random_selection(train_tile, predictions, possible_indices, num_new_patches):
    return random_sample(possible_indices, num_new_patches)

    
def filter_mistakes(train_tile, predictions, possible_indices, find_mistakes=True):
    predicted_classes = np.argmax(predictions, axis=-1)
    # (height, width)
    true_classes = train_tile[0, 4]
    
    mistakes = (predicted_classes != true_classes)
    selectable_points = [point for point in possible_indices if mistakes[point] == find_mistakes]
    return selectable_points


def mistake_random_selection(train_tile, predictions, possible_indices, num_new_patches):
    # predictions: (height, width, channels)
    selectable_mistakes = filter_mistakes(train_tile, predictions, possible_indices)
    return random_selection(train_tile, predictions, selectable_mistakes, num_new_patches)

def mistake_entropy_selection(train_tile, predictions, possible_indices, num_new_patches):
    # predictions: (height, width, channels)
    selectable_mistakes = filter_mistakes(train_tile, predictions, possible_indices)
    return entropy_selection(train_tile, predictions, selectable_mistakes, num_new_patches)

def mistake_margin_selection(train_tile, predictions, possible_indices, num_new_patches):
    # predictions: (height, width, channels)
    selectable_mistakes = filter_mistakes(train_tile, predictions, possible_indices)
    return entropy_selection(train_tile, predictions, selectable_mistakes, num_new_patches)


def half_mistake_selection(train_tile, predictions, possible_indices, num_new_patches):
    # predictions: (height, width, channels)
    selectable_mistakes = filter_mistakes(train_tile, predictions, possible_indices, find_mistakes=True)
    selectable_non_mistakes = filter_mistakes(train_tile, predictions, possible_indices, find_mistakes=False)

    return random_selection(train_tile, predictions, selectable_mistakes, num_new_patches // 2) + \
           random_selection(train_tile, predictions, selectable_non_mistakes, num_new_patches // 2)

    
def new_train_patches(model, train_tile, predictions, num_new_patches, query_strategy=random_selection): # query strategy: random_selection, entropy_selection
    _, _, rows, columns = train_tile.shape

    try:
        margin = model.border_margin_px
    except:
        margin = 0

    margin = max(margin, 240//2)

    # Select from whole tile of non-zero-labeled points
    all_indices = list(zip(*train_tile[0, 4, margin:rows-margin, margin:columns-margin].nonzero()))
    all_indices = [(row + margin, column + margin) for (row, column) in all_indices]
    possible_indices = all_indices
    # Un-comment below for faster debugging
    #num_possible_points = 10000
    #possible_indices = [
    #   (random.randint(margin, rows - margin - 1),
    #    random.randint(margin, columns - margin - 1))
    #    for n in range(num_possible_points)
    #]

    for row, column in possible_indices:
        if not( (margin <= row < rows - margin) and
                (margin <= column < columns - margin) ):
            raise Exception('Invalid point (%d, %d): falls in border of %d px, where a prediction is not possible' % (row, column, margin))      

    selected_points = query_strategy(train_tile, predictions, possible_indices, num_new_patches)
    new_train_patches = pixels_to_patches(train_tile, selected_points)

    return new_train_patches


def run_model(model, naip_data, output_file_path=None):
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

    y_hat = predict_entire_image_unet_fine(model, naip_data)
    # (h, w, c)
    if output_file_path:
        np.save(output_file_path, y_hat)
    class_predictions = np.argmax(y_hat, axis=-1)    

    return y_hat, class_predictions


class ModelState:
    def __init__(self, model, loss, optimizer=None):
        self.model = model
        self.loss = loss
        if optimizer: self.optimizer = optimizer


        
def active_learning_dropout(load_model, loss_criterion, dataloaders, params, params_train, hyper_parameters, log_writer, num_epochs=20, superres=False, masking=True, step_size_function=active_learning_step_size, num_total_points=2000):

    model_state = ModelState(*load_model())
    query_strategy = get_query_strategy(args.active_learning_strategy)
        
    train_tile_fn = open(args.train_tiles_list_file_name, "r").read().strip().split("\n")[0]
    train_tile_fn = train_tile_fn.replace('.mrf', '.npy')
    train_tile = load_tile(train_tile_fn)
    # train_tile: (batch, channels, row, col)
    batch_size, channels, height, width = train_tile.shape
    
    y_train_hr = train_tile[0, 4, :, :]

    train_tile_inputs = features(train_tile)
    train_tile_labels = labels(train_tile)
    
    training_patches = []
    
    try:
        margin = model.border_margin_px
    except:
        margin = 0

    method_and_hypers_str = hyper_parameters_str_dropout(hyper_parameters)

    num_steps = 0
    while len(training_patches) < num_total_points:

        # Evaluate current model
        logits, class_predictions = run_model(model_state.model, train_tile_inputs)
        tile_mean_IoU = mean_IoU(class_predictions[margin:height-margin, margin:width-margin], y_train_hr[margin:height-margin, margin:width-margin], ignored_classes={0})
        tile_pixel_accuracy = pixel_accuracy(class_predictions[margin:height-margin, margin:width-margin], y_train_hr[margin:height-margin, margin:width-margin], ignored_classes={0})
        print('%s, %d, %s, %d, %f, %f, %s' % (method_and_hypers_str, len(training_patches), args.area, args.random_seed, tile_mean_IoU, tile_pixel_accuracy, train_tile_fn))

        # Select new points 
        num_new_patches = step_size_function(num_steps)
        training_patches += new_train_patches(model_state.model, train_tile, logits, num_new_patches, query_strategy=query_strategy)
        training_set = DataGenerator(
            training_patches, params_train["batch_size"], params["patch_size"], params["loader_opts"]["num_channels"], superres=superres)  # superres=params["train_opts"]["superres"]
        dataloaders['train'] = data.DataLoader(training_set, **params_train)

        # Train new model
        model_state.__init__(*load_model())
        hyper_parameters['query_method'] = args.active_learning_strategy
        hyper_parameters['num_points'] = len(training_patches)
        model_state.model, fine_tune_result = train_model_dropout(model_state.model, model_state.loss, dataloaders, hyper_parameters, log_writer, num_epochs=num_epochs, superres=superres, masking=False)

        num_steps += 1

    # Evaluate model after final epoch
    logits, class_predictions = run_model(model_state.model, train_tile_inputs)
    tile_mean_IoU = mean_IoU(class_predictions[margin:height-margin, margin:width-margin], y_train_hr[margin:height-margin, margin:width-margin], ignored_classes={0})
    tile_pixel_accuracy = pixel_accuracy(class_predictions[margin:height-margin, margin:width-margin], y_train_hr[margin:height-margin, margin:width-margin], ignored_classes={0})
    print('%s, %d, %s, %d, %f, %f, %s' % (method_and_hypers_str, len(training_patches), args.area, args.random_seed, tile_mean_IoU, tile_pixel_accuracy, train_tile_fn))


    
def active_learning(load_model, loss_criterion, optimizer, scheduler, dataloaders, params, params_train, hyper_parameters, log_writer, num_epochs=20, superres=False, masking=True, step_size_function=active_learning_step_size, num_total_points=2000):

    model_state = ModelState(*load_model())
    query_strategy = get_query_strategy(args.active_learning_strategy)
        
    train_tile_fn = open(args.train_tiles_list_file_name, "r").read().strip().split("\n")[0]
    train_tile_fn = train_tile_fn.replace('.mrf', '.npy')
    train_tile = load_tile(train_tile_fn)
    # train_tile: (batch, channels, row, col)
    batch_size, channels, height, width = train_tile.shape
    
    y_train_hr = train_tile[0, 4, :, :]

    train_tile_inputs = features(train_tile)
    train_tile_labels = labels(train_tile)
    
    training_patches = []
    
    try:
        margin = model.border_margin_px
    except:
        margin = 0

    method_and_hypers_str = hyper_parameters_str(hyper_parameters)

    num_steps = 0
    while len(training_patches) < num_total_points:
        # Evaluate current model
        logits, class_predictions = run_model(model_state.model, train_tile_inputs)
        tile_mean_IoU = mean_IoU(class_predictions[margin:height-margin, margin:width-margin], y_train_hr[margin:height-margin, margin:width-margin], ignored_classes={0})
        tile_pixel_accuracy = pixel_accuracy(class_predictions[margin:height-margin, margin:width-margin], y_train_hr[margin:height-margin, margin:width-margin], ignored_classes={0})
        print('%s, %d, %s, %d, %f, %f, %s' % (method_and_hypers_str, len(training_patches), args.area, args.random_seed, tile_mean_IoU, tile_pixel_accuracy, train_tile_fn))

        # Select new points 
        num_new_patches = step_size_function(num_steps)
        training_patches += new_train_patches(model_state.model, train_tile, logits, num_new_patches, query_strategy=query_strategy)
        training_set = DataGenerator(
            training_patches, params_train["batch_size"], params["patch_size"], params["loader_opts"]["num_channels"], superres=superres)  # superres=params["train_opts"]["superres"]
        dataloaders['train'] = data.DataLoader(training_set, **params_train)

        # Train new model
        model_state.__init__(*load_model())
        hyper_parameters['query_method'] = args.active_learning_strategy
        hyper_parameters['num_points'] = len(training_patches)
        model_state.model, fine_tune_result = train_model(model_state.model, model_state.loss, model_state.optimizer, scheduler, dataloaders, hyper_parameters, log_writer, num_epochs=num_epochs, superres=superres, masking=False)

        num_steps += 1

    # Evaluate model after final epoch
    logits, class_predictions = run_model(model_state.model, train_tile_inputs)
    tile_mean_IoU = mean_IoU(class_predictions[margin:height-margin, margin:width-margin], y_train_hr[margin:height-margin, margin:width-margin], ignored_classes={0})
    tile_pixel_accuracy = pixel_accuracy(class_predictions[margin:height-margin, margin:width-margin], y_train_hr[margin:height-margin, margin:width-margin], ignored_classes={0})
    print('%s, %d, %s, %d, %f, %f, %s' % (method_and_hypers_str, len(training_patches), args.area, args.random_seed, tile_mean_IoU, tile_pixel_accuracy, train_tile_fn))


def train_model(model, criterion, optimizer, scheduler, dataloaders, hyper_parameters, log_writer, num_epochs=20, superres=False, masking=False):
    global results_writer, results_file
    
    # mask_id indices (points per patch): [1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80, 100]
    mask_id = hyper_parameters['mask_id']
    
    since = datetime.now()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mean_IoU = 0.0
    best_epoch = -1
    duration_til_best_epoch = since - since
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Each epoch can have a training and validation phase
    phases = dataloaders.keys()
    for phase in phases:
        if phase not in ['train', 'val']:
            print('Warning: epoch phase "%s" not valid. Valid options: ["train", "val"]. Data provided in this phase may be ignored.' % phase)

    for epoch in range(-1, num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)
        
        statistics = {
            'mean_IoU': -1,
            'loss': -1,
            'accuracy': -1
        }

        epoch_statistics = {
            phase: copy.deepcopy(statistics)
            for phase in phases
        }
        # print(epoch_statistics)

        hyper_parameters['epoch'] = epoch

        for phase in phases:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:  # phase == 'val'
                if 'val' in dataloaders:
                    model.eval()   # Set model to evaluate mode
                else:
                    continue
            epoch_statistics[phase]['loss'] = 0.0
            epoch_statistics[phase]['mean_IoU'] = 0.0
            epoch_statistics[phase]['accuracy'] = 0.0
            
            running_loss = 0.0
            meanIoU = 0.0
            accuracy = 0.0
            
            n_iter = 0

            # Iterate over data.
            for entry in dataloaders[phase]:
                if superres:
                    if masking:
                        inputs, labels, nlcd, masks = entry
                    else:
                        inputs, labels, nlcd = entry
                    # TODO: use nlcd for superres training, below
                else:
                    if masking:
                        inputs, labels, masks = entry
                    else:
                        inputs, labels = entry

                inputs = inputs[:, :, 2:240 - 2, 2:240 - 2]
                labels = labels[:, :, 94:240 - 94, 94:240 - 94]
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                if masking and phase == 'train':
                    masks = masks.float()
                    masks = masks.to(device)
                    masks = rearrange(masks, 'batch unknown masks height width -> batch (unknown masks) height width')
                    mask = masks[:, mask_id : mask_id + 1, 94:240 - 94, 94:240 - 94].to(device)
                    labels = labels * mask

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train' and epoch > -1):
                    outputs = model.forward(inputs)
                    ground_truth = torch.squeeze(labels,1).long()
                    # print(outputs.shape)
                    # print(ground_truth.shape)
                    path = str(Path(args.model_output_directory) / ("epoch_" + str(epoch) + "_" + phase))
                    ensure_dir(path)
                    # print('Save to path: %s' % path)
                    # save_visualize(inputs, outputs, ground_truth, path)

                    # pdb.set_trace() # print(outputs.shape, ground_truth.shape)
                    loss = criterion(ground_truth, outputs)

                    # backward + optimize only if in training phase
                    if phase == 'train' and epoch > -1:
                        loss.backward()
                        # print(loss.item())
                        optimizer.step()

                # Store ground truth
                y_hr = np.squeeze(labels.cpu().numpy(), axis=1)
                # TODO: I think we need the below... causes error though:
                #if phase == 'train':
                    #y_hr = y_hr * mask.cpu().detach().numpy()
                
                # Store current outputs
                batch_size, _, _ = y_hr.shape
                # TODO: do we need this check below?
                if phase == 'train':
                    y_hat = outputs.cpu().detach().numpy() # * mask.cpu().detach().numpy()
                else:
                    y_hat = outputs.cpu().numpy()
                y_hat = np.argmax(y_hat, axis=1)

                
                # statistics
                n_iter += 1

                # 1) Loss
                epoch_statistics[phase]['loss'] += loss.item()
                
                # 2) mean_IoU
                batch_meanIoU = 0
                for j in range(batch_size):
                    batch_meanIoU += mean_IoU(y_hat[j], y_hr[j], ignored_classes={0})
                batch_meanIoU /= batch_size
                epoch_statistics[phase]['mean_IoU'] += batch_meanIoU
                
                # 3) accuracy
                batch_accuracy = 0
                for j in range(batch_size):
                    batch_accuracy += pixel_accuracy(y_hat[j], y_hr[j], ignored_classes={0})
                batch_accuracy /= batch_size
                epoch_statistics[phase]['accuracy'] += batch_accuracy
                
            # Normalize statistics per training iteration in epoch
            for key in epoch_statistics[phase]:
                epoch_statistics[phase][key] /= n_iter  # divide by how many batches were processed in this epoch

            # print('number of batches in epoch', len(dataloaders[phase]))
            # print('n_iter', n_iter)
                
        result_row = {
            'run_id': hyper_parameters['run_id'],
            'hyper_parameters': hyper_parameters,
            'epoch': epoch,
            'train_loss': epoch_statistics['train']['loss'],
            'train_accuracy': epoch_statistics['train']['accuracy'],
            'train_mean_IoU': epoch_statistics['train']['mean_IoU'],
            'val_loss': epoch_statistics['val']['loss'] if 'val' in epoch_statistics else None,
            'val_accuracy': epoch_statistics['val']['accuracy'] if 'val' in epoch_statistics else None,
            'val_mean_IoU': epoch_statistics['val']['mean_IoU'] if 'val' in epoch_statistics else None,
            'total_time': datetime.now() - since
        }
        #print(result_row)
        results_writer.writerow(result_row)
        results_file.flush()

        model_file_name_suffix = hyper_parameters_str(hyper_parameters)
        
        finetuned_fn = str(Path(args.model_output_directory) / ("finetuned_unet_gn.pth_%s.tar" % model_file_name_suffix))

        torch.save(model.state_dict(), finetuned_fn)
        
        # deep copy the model
            #if phase == 'val' and epoch_mean_IoU > best_mean_IoU:
            #    best_mean_IoU = epoch_mean_IoU
            #    best_model_wts = copy.deepcopy(model.state_dict())
            #    best_epoch = epoch
            #    duration_til_best_epoch = datetime.now() - since
        # print()
        
    # for area, tile_file_names in zip(args.test_areas, args.test_tile_list_file_names):
    # test_model(finetuned_fn, args.config_file, args.area, args.train_tile_list_file_names, 'train')
    # test_model(finetuned_fn, args.config_file, args.area, args.test_tile_list_file_names, 'test')

    duration = datetime.now() - since
    seconds_elapsed = duration.total_seconds()
    
    ## print('Training complete in {:.0f}m {:.0f}s'.format(
    #    seconds_elapsed // 60, seconds_elapsed % 60))
    ## print('Best val IoU: {:4f}'.format(best_mean_IoU))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, FineTuneResult(best_mean_IoU=best_mean_IoU, train_duration=duration)


def hyper_parameters_str(hyper_parameters):
    # hyper_parameters_str = sorted(hyper_parameters.items())
    hyper_parameters_str = "%s_lr_%f" % (
        hyper_parameters['method_name'],
        hyper_parameters['learning_rate'],
    )
    if 'last_k_layers' in hyper_parameters:
        hyper_parameters_str += ("_last_k_" + str(hyper_parameters['last_k_layers']))
    if 'epoch' in hyper_parameters:
        hyper_parameters_str += ("_epoch_" + str(hyper_parameters['epoch']))
    return hyper_parameters_str

def hyper_parameters_str_dropout(hyper_parameters):
    # hyper_parameters_str = sorted(hyper_parameters.items())
    hyper_parameters_str = "%s_lr_%f" % (
        hyper_parameters['method_name'],
        hyper_parameters['dropout_rate'],
    )
    if 'last_k_layers' in hyper_parameters:
        hyper_parameters_str += ("_last_k_" + str(hyper_parameters['last_k_layers']))
    if 'epoch' in hyper_parameters:
        hyper_parameters_str += ("_epoch_" + str(hyper_parameters['epoch']))
    return hyper_parameters_str

def main(finetune_methods, predictions_path, validation_patches_fn=None):
    global results_writer, results_file

    os.makedirs(str(Path(args.log_fn).parent), exist_ok=True)
    
    print(args.log_fn)

    results_file = open(args.log_fn, 'w+')
    results_writer = csv.DictWriter(results_file, ['run_id', 'hyper_parameters', 'epoch', 'train_loss', 'train_accuracy', 'train_mean_IoU', 'val_loss', 'val_accuracy', 'val_mean_IoU', 'total_time'])
    results_writer.writeheader()

    print("method, num_points, area, random_seed, mean_IoU, pixel_accuracy, tile_path")
    
    params = json.load(open(args.config_file, "r"))
    
    validation_patches = None

    # f = open(training_points_sample_fn, "r")
    # training_points = [ for line in f.read().stip().split("\n")]
    
    batch_size = params["loader_opts"]["batch_size"]
    patch_size = params["patch_size"]
    num_channels = params["loader_opts"]["num_channels"]
    params_train = {'batch_size': params["loader_opts"]["batch_size"],
                    'shuffle': params["loader_opts"]["shuffle"],
                    'num_workers': params["loader_opts"]["num_workers"]}
    
    model_opts = params["model_opts"]
    loss = multiclass_ce_points
    path = args.model_file

    dataloaders = {}

    results = {}
    for run_id, (finetune_method_name, finetune_function, hyper_params) in enumerate(finetune_methods):
        hyper_params['run_id'] = run_id
        
        # print('Fine-tune hyper-params: %s' % str(hyper_params))
        improve_reproducibility(args.random_seed)
        model = finetune_function(path, loss, dataloaders, params, params_train, hyper_params, results_writer, n_epochs=hyper_params['n_epochs']) #, predictions_path=str(predictions_path / str(hyper_params)))
        # results[finetune_method_name] = result
        
        savedir = args.model_output_directory
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        # TODO: figure out error, then put this back:
#        if model_opts["model"] == "unet":
#            finetuned_fn = str(Path(savedir) / ("finetuned_unet_gn.pth_%s.tar" % str(hyper_params)))
#            torch.save(model.state_dict(), finetuned_fn)

    # pprint(results)
    # results_file.close()


def hyper_parameters_search(hyper_parameters):
    experiment_configs = []

    num_learning_rates = 6
    num_last_k_layers = 6

    # Random hyper-parameter sweep
    for method_name in ['last_k_layers', 'group_params']:
        for mask_id in [0, 4, 7, 11]:  # 1, 5, 20, or 100 points per patch
            if method_name == 'last_k_layers':
                num_iters = num_learning_rates * num_last_k_layers
            else:
                num_iters = num_learning_rates
            
            for i in range(num_iters):
                hyper_parameters['method_name'] = method_name
                hyper_parameters['mask_id'] = mask_id
                hyper_parameters['learning_rate'] = math.pow(10, random.gauss(-2.55, 0.85))   # random sample centered between 10^-3.4 and 10^-1.7 (0.0004 to 0.02), sampled on log-Gaussian scale
            
                if method_name == 'last_k_layers':
                    fine_tune_function = finetune_last_k_layers
                    hyper_parameters['last_k_layers'] = random.randint(1, 10) # model has 10 layers; in the extreme case, we re-train the whole network...
                else:
                    fine_tune_function = finetune_group_params

                experiment_configs += [(method_name, fine_tune_function, hyper_parameters)]
                hyper_parameters = copy.deepcopy(hyper_parameters)
    return experiment_configs


def hyper_parameters_fixed(hyper_parameters):
    experiment_configs = []

    hyper_parameters_grad_init = {
        'optimizer_method': torch.optim.Adam, #, torch.optim.SGD],
        'learning_rate': 0.004, # [0.001, 0.002, 0.003, 0.004, 0.01, 0.03],
        'lr_schedule_step_size': 1000,  # [5],
        'mask_id': 4, #  [0, 4, 7, 11] # range(12) # [4], # mask-id 5 --> 10 px / patch  # Not currently used
        'n_epochs': 10,
    }
    
    # Add last-k-layers hypers
    for last_k_layers, learning_rate in [(1, 0.01), (2, 0.005), (3, 0.001)]:
        new_hyper_parameters = copy.deepcopy(hyper_parameters_grad_init)
        new_hyper_parameters['method_name'] = 'last_k_layers'
        new_hyper_parameters['last_k_layers'] = last_k_layers
        new_hyper_parameters['learning_rate'] = learning_rate
        experiment_configs += [(new_hyper_parameters['method_name'], finetune_last_k_layers, new_hyper_parameters)]

    # Add group-params method
    new_hyper_parameters = copy.deepcopy(hyper_parameters_grad_init)
    new_hyper_parameters['method_name'] = 'group_params'
    new_hyper_parameters['learning_rate'] = 0.0025
    experiment_configs += [(new_hyper_parameters['method_name'], finetune_group_params, new_hyper_parameters)]

    hyper_parameters_dropout = {
        'mask_id': 0,
	'method_name': 'dropout',
	'n_epochs': 64,
        'last_k_layers': 5,
        'dropout_rate' : 0.2
    }
    
    experiment_configs += [ ('dropout', finetune_dropout, hyper_parameters_dropout) ]
    
    return experiment_configs

    
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

        
if __name__ == "__main__":
    # mask_id indices (points per patch): [1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80, 100]

    hyper_parameters_grad_init = {
        'optimizer_method': torch.optim.Adam, #, torch.optim.SGD],
        'learning_rate': 0.004, # [0.001, 0.002, 0.003, 0.004, 0.01, 0.03],
        'lr_schedule_step_size': 1000,  # [5],
        'mask_id': 4, #  [0, 4, 7, 11] # range(12) # [4], # mask-id 5 --> 10 px / patch  # Not currently used
        'n_epochs': 10,
    }
    
    if args.run_validation:
        experiment_configs = hyper_parameters_search(hyper_parameters_grad_init)
    else:
        experiment_configs = hyper_parameters_fixed(hyper_parameters_grad_init)
    
    
    predictions_path = Path(args.model_output_directory) / "predictions"
    main(experiment_configs, predictions_path)

