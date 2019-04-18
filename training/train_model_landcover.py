#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: skip-file
#
# Copyright © 2017 Caleb Robinson <calebrob6@gmail.com>
#
# Distributed under terms of the MIT license.
'''Runs minibatch sampling algorithms on migration datasets
'''
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

import shutil
import time
import argparse
import datetime

import numpy as np

import utils
import models
import datagen

from keras.callbacks import LearningRateScheduler, ModelCheckpoint

def do_args(arg_list, name):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument("-v", "--verbose", action="store", dest="verbose", type=int, help="Verbosity of keras.fit", default=2)
    parser.add_argument("--output", action="store", dest="output", type=str, help="Output base directory", required=True)
    parser.add_argument("--name", action="store", dest="name", type=str, help="Experiment name", required=True)
    parser.add_argument("--gpu", action="store", dest="gpu", type=int, help="GPU id to use", required=False)

    parser.add_argument("--training_patches", action="store", dest="training_patches_fn", type=str, help="Path to file containing training patches", required=True)
    parser.add_argument("--validation_patches", action="store", dest="validation_patches_fn", type=str, help="Path to file containing validation patches", required=True)

    parser.add_argument("--superres_states", action="store", dest="superres_states", type=str, nargs='+', help="States to use only superres loss with", required=False)


    parser.add_argument("--model_type", action="store", dest="model_type", type=str, \
        choices=["baseline", "extended", "extended_bn", "extended2_bn", "unet1", "unet2", "unet3"], \
        help="Model architecture to use", required=True
    )

    # training arguments
    parser.add_argument("--batch_size", action="store", type=eval, help="Batch size", default="128")
    parser.add_argument("--time_budget", action="store", type=int, help="Time limit", default=3600*3)
    parser.add_argument("--learning_rate", action="store", type=float, help="Learning rate", default=0.003)
    parser.add_argument("--loss", action="store", type=str, help="Loss function", \
        choices=["crossentropy", "jaccard", "superres"], required=True)

    return parser.parse_args(arg_list)


def main():
    prog_name = sys.argv[0]
    args = do_args(sys.argv[1:], prog_name)

    verbose = args.verbose
    output = args.output
    name = args.name

    training_patches_fn = args.training_patches_fn
    validation_patches_fn = args.validation_patches_fn
    superres_states = args.superres_states

    model_type = args.model_type
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    time_budget = args.time_budget
    loss = args.loss


    log_dir = os.path.join(output, name)

    assert os.path.exists(log_dir), "Output directory doesn't exist"

    f = open(os.path.join(log_dir, "args.txt"), "w")
    f.write("%s\n"  % (" ".join(sys.argv)))
    f.close()

    print("Starting %s at %s" % (prog_name, str(datetime.datetime.now())))
    start_time = float(time.time())

    #------------------------------
    # Step 1, load data
    #------------------------------

    f = open(training_patches_fn, "r")
    training_patches = f.read().strip().split("\n")
    f.close()

    f = open(validation_patches_fn, "r")
    validation_patches = f.read().strip().split("\n")
    f.close()

    print("Loaded %d training patches and %d validation patches" % (len(training_patches), len(validation_patches)))

    if loss == "superres":
        print("Using %d states in superres loss:" % (len(superres_states)))
        print(superres_states)

    '''
    highres_patches = []
    for fn in training_patches:
        parts = fn.split("-")
        parts = np.array(list(map(int, parts[2].split("_"))))
        if parts[0] == 0:
            highres_patches.append(fn)
    '''

    #------------------------------
    # Step 2, run experiment
    #------------------------------

    
    #training_steps_per_epoch = len(training_patches) // batch_size // 16
    #validation_steps_per_epoch = len(validation_patches) // batch_size // 16

    training_steps_per_epoch = 300
    validation_steps_per_epoch = 39

    print("Number of training/validation steps per epoch: %d/%d" % (training_steps_per_epoch, validation_steps_per_epoch))


    # Build the model
    if model_type == "baseline":
        model = models.baseline_model_landcover((240,240,4), 5, lr=learning_rate, loss=loss)
    elif model_type == "extended":
        model = models.extended_model_landcover((240,240,4), 5, lr=learning_rate, loss=loss)
    elif model_type == "extended_bn":
        model = models.extended_model_bn_landcover((240,240,4), 5, lr=learning_rate, loss=loss)
    elif model_type == "extended2_bn":
        model = models.extended2_model_bn_landcover((240,240,4), 5, lr=learning_rate, loss=loss)
    elif model_type == "unet1":
        model = models.unet_landcover(
            (240,240,4), out_ch=5, start_ch=64, depth=3, inc_rate=2., activation='relu', 
            dropout=0.5, batchnorm=True, maxpool=True, upconv=True, residual=False, lr=learning_rate, loss=loss
        )
    elif model_type == "unet2":
        model = models.unet_landcover(
            (240,240,4), out_ch=5, start_ch=32, depth=4, inc_rate=2., activation='relu', 
            dropout=False, batchnorm=True, maxpool=True, upconv=True, residual=False, lr=learning_rate, loss=loss
        )
    model.summary()
    

    def schedule_decay(epoch, lr, decay=0.001):
        if epoch>=10:
            lr = lr * 1/(1 + decay * epoch)
        return lr
    
    def schedule_stepped(epoch, lr):
        if epoch < 10:
            return 0.003
        elif epoch < 20:
            return 0.0003
        elif epoch < 30:
            return 0.00015
        else:
            return 0.00003

    validation_callback = utils.LandcoverResults(log_dir=log_dir, time_budget=time_budget, verbose=False)
    learning_rate_callback = LearningRateScheduler(schedule_stepped, verbose=1)
    model_checkpoint_callback = ModelCheckpoint(
        os.path.join(log_dir, "model_{epoch:02d}.h5"),
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        period=1
    )

    training_generator = None
    validation_generator = None
    if loss == "superres":
        training_generator = datagen.DataGenerator(training_patches, batch_size, training_steps_per_epoch, 240, 240, 4, superres=True, superres_states=superres_states)
        validation_generator = datagen.DataGenerator(validation_patches, batch_size, validation_steps_per_epoch, 240, 240, 4, superres=True, superres_states=[])
    else:
        training_generator = datagen.DataGenerator(training_patches, batch_size, training_steps_per_epoch, 240, 240, 4)
        validation_generator = datagen.DataGenerator(validation_patches, batch_size, validation_steps_per_epoch, 240, 240, 4)

    model.fit_generator(
        training_generator,
        steps_per_epoch=training_steps_per_epoch,
        epochs=10**6,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=validation_steps_per_epoch,
        max_queue_size=64,
        workers=4,
        use_multiprocessing=True,
        callbacks=[validation_callback, learning_rate_callback, model_checkpoint_callback],
        initial_epoch=0 
    )

    #------------------------------
    # Step 3, save models
    #------------------------------
    model.save(os.path.join(log_dir, "final_model.h5"))

    model_json = model.to_json()
    with open(os.path.join(log_dir,"final_model.json"), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(log_dir, "final_model_weights.h5"))

    print("Finished in %0.4f seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()