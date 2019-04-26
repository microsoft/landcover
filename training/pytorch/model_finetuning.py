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

import torch
import torch.nn as nn
import json
from training.pytorch.models.unet import Unet
from training.pytorch.models.fusionnet import Fusionnet
from torch.optim import lr_scheduler
import copy
from training.pytorch.utils.eval_segm import mean_IoU
from training.pytorch.utils.experiments_utils import improve_reproducibility
from training.pytorch.losses import (multiclass_ce, multiclass_dice_loss, multiclass_jaccard_loss, multiclass_tversky_loss, multiclass_ce_points)
from training.pytorch.data_loader import DataGenerator
from torch.utils import data
import os


parser = argparse.ArgumentParser()

parser.add_argument('--config_file', type=str, default="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/params.json", help="json file containing the configuration")

parser.add_argument('--model_file', type=str,
                    help="Checkpoint saved model",
                    default="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar")

parser.add_argument('--data_path', type=str, help="Path to data", default="/mnt/blobfuse/cnn-minibatches/summer_2019/active_learning_splits/")
parser.add_argument('--data_sub_dirs', type=str, nargs='+', help="Sub-directories of `data_path` to get data from", default=['val1',]) # 'test1', 'test2', 'test3', 'test4'])
parser.add_argument('--validation_patches_fn', type=str, help="Filename with list of validation patch files", default='training/data/finetuning/val1_test_patches_500.txt')
parser.add_argument('--training_patches_fn', type=str, help="Filename with list of training patch files", default="training/data/finetuning/val1_train_patches.txt")
parser.add_argument('--log_fn', type=str, help="Where to store training results", default="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/finetune_results.csv")


args = parser.parse_args()

class GroupParams(nn.Module):

    def __init__(self, model):
        super(GroupParams, self).__init__()
        self.gammas = nn.Parameter(torch.ones((1, 32, 1, 1)))
        self.betas = nn.Parameter(torch.zeros((1, 32, 1, 1)))
        self.model = model

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
    
    
def finetune_group_params(path_2_saved_model, loss, gen_loaders, params, hyper_parameters, log_writer, n_epochs=25):
    learning_rate = hyper_parameters['learning_rate']
    optimizer_method = hyper_parameters['optimizer_method']
    lr_schedule_step_size = hyper_parameters['lr_schedule_step_size']
    
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
    loss = loss().to(device)

    optimizer = torch.optim.SGD(model_2_finetune.parameters(), lr=learning_rate, momentum=0.9)
    if optimizer_method == torch.optim.Adam:
        optimizer = torch.optim.Adam(model_2_finetune.parameters(), lr=learning_rate, eps=1e-5)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_schedule_step_size, gamma=0.1)

    model_2_finetune = train_model(model_2_finetune, loss, optimizer,
                                   exp_lr_scheduler, gen_loaders, hyper_parameters, log_writer, num_epochs=n_epochs)
    return model_2_finetune

def finetune_last_k_layers(path_2_saved_model, loss, gen_loaders, params, hyper_parameters, log_writer, n_epochs=25):
    learning_rate = hyper_parameters['learning_rate']
    optimizer_method = hyper_parameters['optimizer_method']
    lr_schedule_step_size = hyper_parameters['lr_schedule_step_size']
    last_k_layers = hyper_parameters['last_k_layers']
    
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
    loss = loss().to(device)

    optimizer = torch.optim.SGD(model_2_finetune.parameters(), lr=learning_rate, momentum=0.9)
    if optimizer_method == torch.optim.Adam:
        optimizer = torch.optim.Adam(model_2_finetune.parameters(), lr=learning_rate, eps=1e-5)
        
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_schedule_step_size, gamma=0.1)

    model_2_finetune = train_model(model_2_finetune, loss, optimizer,
                                   exp_lr_scheduler, gen_loaders, hypers, log_writer, num_epochs=n_epochs)
    return model_2_finetune


def train_model(model, criterion, optimizer, scheduler, dataloaders, hyper_parameters, log_writer, num_epochs=5, superres=False, masking=True):
    global results_writer
    
    # mask_id indices (points per patch): [1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80, 100]
    mask_id = 11
    
    since = datetime.now()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mean_IoU = 0.0
    best_epoch = -1
    duration_til_best_epoch = since - since
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:  # phase == 'val'
                if 'val' in dataloaders:
                    model.eval()   # Set model to evaluate mode
                else:
                    continue

            running_loss = 0.0
            meanIoU = 0.0
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
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs)
                    loss = criterion(torch.squeeze(labels,1).long(), outputs)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                n_iter+=1
                #if phase == 'val':
                y_hr = np.squeeze(labels.cpu().numpy(), axis=1)
                batch_size, _, _ = y_hr.shape
                # TODO: do we need this check below?
                if phase == 'train':
                    y_hat = outputs.cpu().detach().numpy() * mask.cpu().detach().numpy()
                else:
                    y_hat = outputs.cpu().numpy()
                y_hat = np.argmax(y_hat, axis=1)
                batch_meanIoU = 0
                for j in range(batch_size):
                    #pdb.set_trace()
                    batch_meanIoU += mean_IoU(y_hat[j], y_hr[j], ignored_classes={0})
                batch_meanIoU /= batch_size
                meanIoU += batch_meanIoU
                # print('batch_meanIoU: %f' % batch_meanIoU)

                if phase == 'val':
                    val_loss = running_loss / n_iter
                    val_mean_IoU = meanIoU / n_iter
                elif phase == 'train':
                    train_loss = running_loss / n_iter
                    tran_mean_IoU = meanIoU / n_iter

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #      phase, epoch_loss, epoch_mean_IoU))
            result_row = {
                'run_id': hyper_parameters['run_id'],
                'hyper_parameters': hyper_parameters,
                'epoch': epoch,
                'train_IoU': train_mean_IoU,
                'train_loss': train_loss,
                'val_IoU': val_mean_IoU,
                'val_loss': val_loss,
                'total_time': datetime.now() - since
            }
            print(result_row)
            result_writer.writerow(result_row)

            # deep copy the model
            #if phase == 'val' and epoch_mean_IoU > best_mean_IoU:
            #    best_mean_IoU = epoch_mean_IoU
            #    best_model_wts = copy.deepcopy(model.state_dict())
            #    best_epoch = epoch
            #    duration_til_best_epoch = datetime.now() - since
        print()

    duration = datetime.now() - since
    seconds_elapsed = duration.total_seconds()
    
    print('Training complete in {:.0f}m {:.0f}s'.format(
        seconds_elapsed // 60, seconds_elapsed % 60))
    print('Best val IoU: {:4f}'.format(best_mean_IoU))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, FineTuneResult(best_mean_IoU=best_mean_IoU, train_duration=duration)

def main(finetune_methods, validation_patches_fn=None):
    global results_writer
    results_file = open(args.log_fn, 'w+')
    results_writer = csv.DictWriter(results_file, ['run_id', 'hyper_parameters', 'epoch', 'train_IoU', 'train_loss', 'val_IoU', 'val_loss', 'total_time'])
    results_writer.writeheader()
    
    params = json.load(open(args.config_file, "r"))
    
    f = open(args.training_patches_fn, "r")
    training_patches = f.read().strip().split("\n")
    f.close()

    validation_patches = None
    if args.validation_patches_fn:
        f = open(args.validation_patches_fn, "r")
        validation_patches = f.read().strip().split("\n")
        f.close()

    # f = open(training_points_sample_fn, "r")
    # training_points = [ for line in f.read().stip().split("\n")]
    
    batch_size = params["loader_opts"]["batch_size"]
    patch_size = params["patch_size"]
    num_channels = params["loader_opts"]["num_channels"]
    params_train = {'batch_size': params["loader_opts"]["batch_size"],
                    'shuffle': params["loader_opts"]["shuffle"],
                    'num_workers': params["loader_opts"]["num_workers"]}
        
    training_set = DataGenerator(
        training_patches, batch_size, patch_size, num_channels, superres=params["train_opts"]["superres"], masking=True
    )

    validation_set = None
    if validation_patches:
        validation_set = DataGenerator(
            validation_patches, batch_size, patch_size, num_channels, superres=params["train_opts"]["superres"], masking=True
        )

    model_opts = params["model_opts"]
    loss = multiclass_ce_points
    path = args.model_file

    dataloaders = {'train': data.DataLoader(training_set, **params_train)}
    if validation_set:
        dataloaders['val'] = data.DataLoader(validation_set, **params_train)

    results = {}
    for run_id, (finetune_method_name, finetune_function, hyper_params) in enumerate(finetune_methods):
        hyper_params['run_id'] = run_id
        print('Fine-tune hyper-params: %s' % str(hyper_params))
        improve_reproducibility()
        model, result = finetune_function(path, loss, dataloaders, params, hyper_params, results_writer, n_epochs=5)
        results[finetune_method_name] = result
        
        savedir = "/mnt/blobfuse/train-output/conditioning/models/finetuning/"
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        
        if model_opts["model"] == "unet":
            finetunned_fn = savedir + "finetuned_unet_gn.pth_%s.tar" % str(fine_tune_params)
            torch.save(model.state_dict(), finetunned_fn)

    pprint(results)
    close(results_file)

    
def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

        
if __name__ == "__main__":
    params_sweep_last_k = {
        'optimizer_method': [torch.optim.Adam], #, torch.optim.SGD],
        'last_k_layers': [1,], #2, 4, 8],
        'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001],
        'lr_schedule_step_size': [7],
    }

    params_sweep_group_norm = {
        'optimizer_method': [torch.optim.Adam], #, torch.optim.SGD],
        'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001],
        'lr_schedule_step_size': [7],
    }

    params_list_last_k = list(product_dict(**params_sweep_last_k))
    params_list_group_norm = list(product_dict(**params_sweep_group_norm))
    
    main([('Group params', finetune_group_params, hypers) for hypers in params_list_group_norm] + \
         [('Last k layers', finetune_last_k_layers, hypers) for hypers in params_list_last_k])
