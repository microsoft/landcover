import argparse
import numpy as np
from pprint import pprint
from functools import partial
from attr import attrs, attrib
from datetime import datetime, timedelta
from pathlib import Path

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
parser.add_argument('--validation_patches_fn', type=str, help="Filename with list of validation patch files", default='')
parser.add_argument('--training_patches_fn', type=str, help="Filename with list of training patch files", default="training/data/finetuning/val1_train_patches.txt")


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
    best_accuracy = attrib(type=float)
    train_duration = attrib(type=timedelta)
    
    
def finetune_group_params(path_2_saved_model, loss, gen_loaders,params, n_epochs=25):
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


    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer = torch.optim.Adam(model_2_finetune.parameters(), lr=0.01, eps=1e-5)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model_2_finetune = train_model(model_2_finetune, loss, optimizer,
                             exp_lr_scheduler, gen_loaders, num_epochs=n_epochs)
    return model_2_finetune

def finetune_last_k_layers(path_2_saved_model, loss, gen_loaders, params, n_epochs=25, last_k_layers=3, learning_rate=0.005, optimizer_method=torch.optim.SGD):
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


    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    if optimizer_method == torch.optim.SGD:
        optimizer = torch.optim.SGD(model_2_finetune.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_method == torch.optim.Adam:
        optimizer = torch.optim.Adam(model_2_finetune.parameters(), lr=learning_rate, eps=1e-5)
    else:
        optimizer = torch.optim.SGD(model_2_finetune.parameters(), lr=learning_rate, momentum=0.9)
        
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model_2_finetune = train_model(model_2_finetune, loss, optimizer,
                             exp_lr_scheduler, gen_loaders, num_epochs=n_epochs)
    return model_2_finetune


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25, superres=False, mask_id=0):
    # mask_id defaults (points per patch): [1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80, 100]
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
            else:
                if 'val' in dataloaders:
                    model.eval()   # Set model to evaluate mode
                else:
                    continue

            running_loss = 0.0
            val_meanIoU = 0.0
            n_iter = 0

            # Iterate over data.
            for entry in dataloaders[phase]:
                if superres:
                    inputs, labels, nlcd, masks = entry
                    # TODO: use nlcd for superres training, below
                else:
                    inputs, labels, masks = entry

                import pdb
                pdb.set_trace()
                
                inputs = inputs[:, :, 2:240 - 2, 2:240 - 2]
                labels = labels[:, :, 94:240 - 94, 94:240 - 94]
                inputs = inputs.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                labels = labels * masks[:, mask_id, 94:240 - 94, 94:240 - 94]

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
                y_hat = outputs.cpu().numpy()
                y_hat = np.argmax(y_hat, axis=1)
                batch_meanIoU = 0
                for j in range(batch_size):
                    batch_meanIoU += mean_IoU(y_hat[j], y_hr[j])
                batch_meanIoU /= batch_size
                val_meanIoU += batch_meanIoU

            epoch_loss = running_loss / n_iter
            epoch_mean_IoU = val_meanIoU / n_iter

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_mean_IoU))

            # deep copy the model
            if phase == 'val' and epoch_mean_IoU > best_mean_IoU:
                best_mean_IoU = epoch_mean_IoU
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                duration_til_best_epoch = datetime.now() - since
        print()

    duration = datetime.now() - since
    seconds_elapsed = duration.total_seconds()
    print('Training complete in {:.0f}m {:.0f}s'.format(
        seconds_elapsed // 60, seconds_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, FineTuneResult(best_accuracy=best_acc, train_duration=duration)

def main(finetune_methods, validation_patches_fn=None):
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
        training_patches, batch_size, patch_size, num_channels, superres=params["train_opts"]["superres"]
    )

    validation_set = None
    if validation_patches:
        validation_set = DataGenerator(
            validation_patches, batch_size, patch_size, num_channels, superres=params["train_opts"]["superres"]
        )

    #train_opts = params["train_opts"]
    model_opts = params["model_opts"]

    # Default model is Duke_Unet


    #if train_opts["loss"] == "dice":
    #    loss = multiclass_dice_loss
    #elif train_opts["loss"] == "ce":
    loss = multiclass_ce_points
    #elif train_opts["loss"] == "jaccard":
    #    loss = multiclass_jaccard_loss
    #elif train_opts["loss"] == "tversky":
    #    loss = multiclass_tversky_loss
    #else:
    #    print("Option {} not supported. Available options: dice, ce, jaccard, tversky".format(train_opts["loss"]))
    #    raise NotImplementedError

    path = args.model_file

    dataloaders = {'train': data.DataLoader(training_set, **params_train)}
    if validation_set:
        dataloaders['val'] = data.DataLoader(validation_set, **params_train)

    results = {}
    for (finetune_method_name, finetune_function) in finetune_methods:
        improve_reproducibility()
        model, result = finetune_function(path, loss, dataloaders, params, n_epochs=10)
        results[finetune_method_name] = result
        
        savedir = "/mnt/blobfuse/train-output/conditioning/models/finetuning/"
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        if model_opts["model"] == "unet":
            finetunned_fn = savedir + "finetuned_unet_gn.pth.tar"
            torch.save(model.state_dict(), finetunned_fn)

    pprint(results)
            
if __name__ == "__main__":
    main([
        ('Group params', finetune_group_params),
        #('SGD on last 1 layers', partial(finetune_last_k_layers, optimizer_method=torch.optim.SGD, last_k_layers=1)),
        ('Adam on last 1 layers', partial(finetune_last_k_layers, optimizer_method=torch.optim.Adam, last_k_layers=1)),
        #('SGD on last 2 layers', partial(finetune_last_k_layers, optimizer_method=torch.optim.SGD, last_k_layers=2)),
        ('Adam on last 2 layers', partial(finetune_last_k_layers, optimizer_method=torch.optim.Adam, last_k_layers=2)),
        #('SGD on last 4 layers', partial(finetune_last_k_layers, optimizer_method=torch.optim.SGD, last_k_layers=4)),
        ('Adam on last 4 layers', partial(finetune_last_k_layers, optimizer_method=torch.optim.Adam, last_k_layers=4)),
    ])
