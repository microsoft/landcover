import torch
import os, sys
import numpy as np
import json
import torch.nn as nn
from time import time
from pytorch.utils.eval_segm import mean_IoU
from torch.optim import lr_scheduler
from pytorch.utils.experiments_utils import (saveLoss, CheckpointSaver, improve_reproducibility, NamespaceFromDict)


class Framework:
    def __init__(self, model, loss, lr=1e-3):
        self.model = model
        self.loss = loss
        self.lr = lr
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        if torch.cuda.is_available():
            self.model = model.cuda()
            self.loss = loss.cuda()

    def optimize(self, X, y):
        y_pred = self.model.forward(X)
        loss = self.loss(y, y_pred)
        return loss, y_pred

    def backwardpass(self, loss):
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

def train_superres(framework, gen_loaders, n_epochs, params):

    if params["checkpoint_file"] == "none":
        resume_run = False

        experiment_dir = os.path.join(params["save_dir"], params["experiment_name"])
        train_dir = os.path.join(experiment_dir, 'training/')

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if os.listdir(train_dir):
            raise IOError('train_dir is expected to be empty for new experiments. '
                          'train_dir is not empty! Aborting...')

        # determine backup folder path and create it if necessary
        backup_dir = os.path.join(
            params["backup_dir"],
            os.path.split(os.path.normpath(train_dir))[1]
        )
        # create the backup dir for storing validation best snapshots
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        # save command-line arguments to train and backup dir
        json.dump(params, open(os.path.join(train_dir, 'params.json'), 'w'),
                  indent=4, sort_keys=True)
        json.dump(params, open(os.path.join(backup_dir, 'params.json'), 'w'),
                  indent=4, sort_keys=True)

        log_file = os.path.join(train_dir, "log_file.log")
        my_log = open(log_file, 'w')
        my_log.write("Starting training")
        my_log.close()


    else:
        resume_run = True
        train_dir = os.path.split(params["checkpoint_file"])[0]

        # load checkpoint overwriting other args
        checkpoint_file = params["checkpoint_file"].checkpoint_file  # backup checkpoint file path
        params = json.load(open(os.path.join(train_dir, 'params.json')))
        backup_dir = os.path.join(params["backup_dir"],
                                  os.path.split(os.path.normpath(train_dir))[1])
        log_file = os.path.join(train_dir, "log_file.log")

    config = NamespaceFromDict(params)
#    improve_reproducibility(seed=config.rand_seed)

    train_opts = params["train_opts"]
    if train_opts["verbose"]:
        print('using training directory {}',train_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    framework.model.to(device)

    if train_opts["parallelize"]:
        device_ids = list(range(torch.cuda.device_count()))
        framework.model = nn.DataParallel(framework.model, device_ids=device_ids)

    tic = time()
    n_early_stopping = 0
    val_history = {'loss': [], 'mean_IoU': []}
    train_history = {'loss': []}

    scheduler = lr_scheduler.StepLR(framework.optimizer, step_size=train_opts["scheduler_step_size"], gamma=train_opts["scheduler_gamma"])

    stats = {
        'train_losses': [], 'train_losses_epochs': [],
        'val_losses': [], 'val_ious': [], 'val_ious_epochs': [],
        'best_val_iou': 0., 'best_val_epoch': 0.,
        'resume_epochs': []
    }

    if resume_run:
        if train_opts["verbose"]:
            print('resuming training...')

        # load epoch, step, state_dict of model, optimizer as well as best val
        # acc and step
        checkpoint = torch.load(checkpoint_file)
        resume_epoch = checkpoint['epoch']
        framework.model.load_state_dict(checkpoint['model'])
        framework.optimizer.load_state_dict(checkpoint['optimizer'])

        stats['train_losses'] = checkpoint['train_losses']
        stats['train_losses_epochs'] = checkpoint['train_losses_epochs']
        stats['val_losses'] = checkpoint['val_losses']
        stats['val_ious'] = checkpoint['val_ious']
        stats['val_ious_epochs'] = checkpoint['val_ious_epochs']
        stats['best_val_iou'] = checkpoint['best_val_iou']
        stats['best_val_epoch'] = checkpoint['best_val_epoch']
        stats['resume_epochs'] = checkpoint['resume_epochs']
        stats['resume_epochs'].append(resume_epoch)


    else:
        if train_opts["verbose"]:
            print('starting training!')
        resume_epoch = 0

    saver = CheckpointSaver(save_dir=train_dir,
                            backup_dir=backup_dir)

    for i in range(resume_epoch, n_epochs):
        my_log = open(log_file, 'a+')
        my_log.write("\n***************")
        if train_opts["verbose"]:
            print("**************")
        for phase in ['train', 'val']:
            epoch_loss = 0.0
            val_loss =0.0
            val_meanIoU = 0.0
            n_iter = 0
            if phase == 'train':
                framework.model.train()
            else:
                framework.model.eval()
            for (X, y_hr, y_sr) in gen_loaders[phase]():
                if torch.cuda.is_available():
                    X = X.cuda()
                    y_hr = y_hr.cuda()
                    y_sr=y_sr.cuda()
                framework.optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred, y_sr_pred = framework.model.forward(X)
                    loss_segm = framework.loss(torch.squeeze(y_hr,1).long(), y_pred)
                    loss_aux = framework.loss(torch.squeeze(y_sr, 1).long(), y_sr_pred)
                    loss = loss_segm + params["train_opts"]["alpha_superres_loss"] * loss_aux

                    if phase == 'train':
                        loss.backward()
                        framework.optimizer.step()
                n_iter += 1
                epoch_loss += loss.item()
                if phase == 'val':
                    y_hr = np.squeeze(y_hr.cpu().numpy(), axis=1)
                    batch_size, _, _ = y_hr.shape
                    y_hat = y_pred.cpu().numpy()
                    y_hat = np.argmax(y_hat, axis=1)
                    batch_meanIoU=0
                    for j in range(batch_size):
                        batch_meanIoU += mean_IoU(y_hat[j], y_hr[j])
                    batch_meanIoU /= batch_size
                    val_meanIoU += batch_meanIoU

            # save if notice improvement
            epoch_loss /= n_iter
            loss_str = "\n{} loss: {} Epoch number: {} time: {}".format(
                str(phase),
                str(epoch_loss),
                str(i),
                str(int(time() - tic))
            )
            my_log = open(log_file, 'a+')
            my_log.write(loss_str)
            if train_opts["verbose"]:
                print(loss_str)

            if phase == 'train':
                stats['train_losses'].append(epoch_loss)
                stats['train_losses_epochs'].append(i)
                train_history['loss'].append(epoch_loss)
            else:
                val_meanIoU /= n_iter
                val_history['loss'].append(epoch_loss)
                val_history['mean_IoU'].append(val_meanIoU)
                stats['val_losses'].append(epoch_loss)
                stats['val_ious_epochs'].append(i)

                scheduler.step(val_loss)
                text = "\nVal meanIoU: {}".format(str(val_meanIoU))
                my_log = open(log_file, 'a+')
                my_log.write(text)
                if train_opts["verbose"]:
                    print("Val meanIoU: " + str(val_meanIoU))
                is_best = False
                if val_meanIoU > stats['best_val_iou']:
                    n_early_stopping = 0
                    is_best = True
                    stats['best_val_iou'] = val_meanIoU
                    stats['best_val_epoch'] = i

                else:
                    n_early_stopping += 1

                checkpoint = {
                    'params': params,
                    'epoch': i,
                    'model': framework.model.state_dict(),
                    'optimizer': framework.optimizer.state_dict(),

                }
                for k, v in stats.items():
                    checkpoint[k] = v
                saver.save(state=checkpoint, is_best=is_best,
                           checkpoint_name='checkpoint')

            if train_opts["early_stopping"] and n_early_stopping > train_opts["early_stopping_patience"]:
                text = "\nEarly stopping on epoch: " + str(i)
                my_log = open(log_file, 'a+')
                my_log.write(text)
                if (train_opts["verbose"]):
                    print("Early stopping on epoch: " + str(i))
                break
        sys.stdout.flush()
    if train_opts["verbose"]:
        print("Done training")
    my_log = open(log_file, 'a+')
    my_log.write("\nDone training")
    saveLoss(train_history['loss'], val_history['loss'], backup_dir, "loss_figure")
    saveLoss(train_history['loss'], val_history['loss'], train_dir, "loss_figure")
    return framework, train_history, val_history

def train(framework, gen_loaders, n_epochs, params):

    if params["checkpoint_file"] == "none":
        resume_run = False

        experiment_dir = os.path.join(params["save_dir"], params["experiment_name"])
        train_dir = os.path.join(experiment_dir, 'training/')

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if os.listdir(train_dir):
            raise IOError('train_dir is expected to be empty for new experiments. '
                          'train_dir is not empty! Aborting...')

        # determine backup folder path and create it if necessary
        backup_dir = os.path.join(
            params["backup_dir"],
            os.path.split(os.path.normpath(train_dir))[1]
        )
        # create the backup dir for storing validation best snapshots
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        # save command-line arguments to train and backup dir
        json.dump(params, open(os.path.join(train_dir, 'params.json'), 'w'),
                  indent=4, sort_keys=True)
        json.dump(params, open(os.path.join(backup_dir, 'params.json'), 'w'),
                  indent=4, sort_keys=True)

        log_file = os.path.join(train_dir, "log_file.log")
        my_log = open(log_file, 'w')
        my_log.write("Starting training")
        my_log.close()


    else:
        resume_run = True
        train_dir = os.path.split(params["checkpoint_file"])[0]

        # load checkpoint overwriting other args
        checkpoint_file = params["checkpoint_file"].checkpoint_file  # backup checkpoint file path
        params = json.load(open(os.path.join(train_dir, 'params.json')))
        backup_dir = os.path.join(params["backup_dir"],
                                  os.path.split(os.path.normpath(train_dir))[1])
        log_file = os.path.join(train_dir, "log_file.log")

    config = NamespaceFromDict(params)
#    improve_reproducibility(seed=config.rand_seed)

    train_opts = params["train_opts"]
    if train_opts["verbose"]:
        print('using training directory {}',train_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    framework.model.to(device)

    if train_opts["parallelize"]:
        device_ids = list(range(torch.cuda.device_count()))
        framework.model = nn.DataParallel(framework.model, device_ids=device_ids)

    tic = time()
    n_early_stopping = 0
    val_history = {'loss': [], 'mean_IoU': []}
    train_history = {'loss': []}

    scheduler = lr_scheduler.StepLR(framework.optimizer, step_size=train_opts["scheduler_step_size"], gamma=train_opts["scheduler_gamma"])

    stats = {
        'train_losses': [], 'train_losses_epochs': [],
        'val_losses': [], 'val_ious': [], 'val_ious_epochs': [],
        'best_val_iou': 0., 'best_val_epoch': 0.,
        'resume_epochs': []
    }

    if resume_run:
        if train_opts["verbose"]:
            print('resuming training...')

        # load epoch, step, state_dict of model, optimizer as well as best val
        # acc and step
        checkpoint = torch.load(checkpoint_file)
        resume_epoch = checkpoint['epoch']
        framework.model.load_state_dict(checkpoint['model'])
        framework.optimizer.load_state_dict(checkpoint['optimizer'])

        stats['train_losses'] = checkpoint['train_losses']
        stats['train_losses_epochs'] = checkpoint['train_losses_epochs']
        stats['val_losses'] = checkpoint['val_losses']
        stats['val_ious'] = checkpoint['val_ious']
        stats['val_ious_epochs'] = checkpoint['val_ious_epochs']
        stats['best_val_iou'] = checkpoint['best_val_iou']
        stats['best_val_epoch'] = checkpoint['best_val_epoch']
        stats['resume_epochs'] = checkpoint['resume_epochs']
        stats['resume_epochs'].append(resume_epoch)


    else:
        if train_opts["verbose"]:
            print('starting training!')
        resume_epoch = 0

    saver = CheckpointSaver(save_dir=train_dir,
                            backup_dir=backup_dir)

    for i in range(resume_epoch, n_epochs):
        my_log = open(log_file, 'a+')
        my_log.write("\n***************")
        if train_opts["verbose"]:
            print("**************")
        for phase in ['train', 'val']:
            epoch_loss = 0.0
            val_loss =0.0
            val_meanIoU = 0.0
            n_iter = 0
            if phase == 'train':
                framework.model.train()
            else:
                framework.model.eval()
            for (X, y_sr) in gen_loaders[phase]():
                if params["model_opts"]["model"] == "unet":
                    X = X[:, :, 2:params["patch_size"] - 2, 2:params["patch_size"] - 2]
                    y_sr = y_sr[:, :, 94:params["patch_size"] - 94, 94:params["patch_size"] - 94]
                if torch.cuda.is_available():
                    X = X.cuda()
                    y_sr = y_sr.cuda()
                framework.optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    y_pred = framework.model.forward(X)
                    loss = framework.loss(torch.squeeze(y_sr,1).long(), y_pred)

                    if phase == 'train':
                        loss.backward()
                        framework.optimizer.step()
                n_iter += 1

                if train_opts["verbose"]:
                    iter_str = "\n{} loss: {} Epoch number: {} time: {} iter: {}".format(
                        str(phase),
                        str(loss.item()),
                        str(i),
                        str(int(time() - tic)),
                        str(n_iter)
                    )
                    print(iter_str)
                epoch_loss += loss.item()
                if phase == 'val':
                    y_sr = np.squeeze(y_sr.cpu().numpy(), axis=1)
                    batch_size, _, _ = y_sr.shape
                    y_hat = y_pred.cpu().numpy()
                    y_hat = np.argmax(y_hat, axis=1)
                    batch_meanIoU=0
                    for j in range(batch_size):
                        batch_meanIoU += mean_IoU(y_hat[j], y_sr[j])
                    batch_meanIoU /= batch_size
                    val_meanIoU += batch_meanIoU

            # save if notice improvement
            epoch_loss /= n_iter
            loss_str = "\n{} loss: {} Epoch number: {} time: {}".format(
                str(phase),
                str(epoch_loss),
                str(i),
                str(int(time() - tic))
            )
            my_log = open(log_file, 'a+')
            my_log.write(loss_str)
            if train_opts["verbose"]:
                print(loss_str)

            if phase == 'train':
                stats['train_losses'].append(epoch_loss)
                stats['train_losses_epochs'].append(i)
                train_history['loss'].append(epoch_loss)
            else:
                val_meanIoU /= n_iter
                val_history['loss'].append(epoch_loss)
                val_history['mean_IoU'].append(val_meanIoU)
                stats['val_losses'].append(epoch_loss)
                stats['val_ious_epochs'].append(i)

                scheduler.step(val_loss)
                text = "\nVal meanIoU: {}".format(str(val_meanIoU))
                my_log = open(log_file, 'a+')
                my_log.write(text)
                if train_opts["verbose"]:
                    print("Val meanIoU: " + str(val_meanIoU))
                is_best = False
                if val_meanIoU > stats['best_val_iou']:
                    n_early_stopping = 0
                    is_best = True
                    stats['best_val_iou'] = val_meanIoU
                    stats['best_val_epoch'] = i

                else:
                    n_early_stopping += 1

                checkpoint = {
                    'params': params,
                    'epoch': i,
                    'model': framework.model.state_dict(),
                    'optimizer': framework.optimizer.state_dict(),

                }
                for k, v in stats.items():
                    checkpoint[k] = v
                saver.save(state=checkpoint, is_best=is_best,
                           checkpoint_name='checkpoint')

            if train_opts["early_stopping"] and n_early_stopping > train_opts["early_stopping_patience"]:
                text = "\nEarly stopping on epoch: " + str(i)
                my_log = open(log_file, 'a+')
                my_log.write(text)
                if (train_opts["verbose"]):
                    print("Early stopping on epoch: " + str(i))
                break
        sys.stdout.flush()
    if train_opts["verbose"]:
        print("Done training")
    my_log = open(log_file, 'a+')
    my_log.write("\nDone training")
    saveLoss(train_history['loss'], val_history['loss'], backup_dir, "loss_figure")
    saveLoss(train_history['loss'], val_history['loss'], train_dir, "loss_figure")
    return framework, train_history, val_history

