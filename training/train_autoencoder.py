#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1137,E1136,E0110
import sys
import os
import time
import collections
import argparse

import numpy as np

import rasterio

import tensorflow.keras.backend as K
import tensorflow.keras.callbacks
import tensorflow.keras.utils
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adadelta
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, MaxPooling2D, Conv2D, BatchNormalization
from tensorflow.keras.layers import Concatenate, Cropping2D, Lambda
from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import LeakyReLU

from sklearn.cluster import MiniBatchKMeans, KMeans

# from https://gist.github.com/seberg/3866040
def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """Create a view of `array` which for every point gives the n-dimensional
    neighbourhood of size window. New dimensions are added at the end of
    `array` or after the corresponding original dimension.
    
    Parameters
    ----------
    array : array_like
        Array to which the rolling window is applied.
    window : int or tuple
        Either a single integer to create a window of only the last axis or a
        tuple to create it for the last len(window) axes. 0 can be used as a
        to ignore a dimension in the window.
    asteps : tuple
        Aligned at the last axis, new steps for the original array, ie. for
        creation of non-overlapping windows. (Equivalent to slicing result)
    wsteps : int or tuple (same size as window)
        steps for the added window dimensions. These can be 0 to repeat values
        along the axis.
    axes: int or tuple
        If given, must have the same size as window. In this case window is
        interpreted as the size in the dimension given by axes. IE. a window
        of (2, 1) is equivalent to window=2 and axis=-2.       
    toend : bool
        If False, the new dimensions are right after the corresponding original
        dimension, instead of at the end of the array. Adding the new axes at the
        end makes it easier to get the neighborhood, however toend=False will give
        a more intuitive result if you view the whole array.
    
    Returns
    -------
    A view on `array` which is smaller to fit the windows and has windows added
    dimensions (0s not counting), ie. every point of `array` is an array of size
    window.
    
    Examples
    --------
    >>> a = np.arange(9).reshape(3,3)
    >>> rolling_window(a, (2,2))
    array([[[[0, 1],
             [3, 4]],
            [[1, 2],
             [4, 5]]],
           [[[3, 4],
             [6, 7]],
            [[4, 5],
             [7, 8]]]])
    
    Or to create non-overlapping windows, but only along the first dimension:
    >>> rolling_window(a, (2,0), asteps=(2,1))
    array([[[0, 3],
            [1, 4],
            [2, 5]]])
    
    Note that the 0 is discared, so that the output dimension is 3:
    >>> rolling_window(a, (2,0), asteps=(2,1)).shape
    (1, 3, 2)
    
    This is useful for example to calculate the maximum in all (overlapping)
    2x2 submatrixes:
    >>> rolling_window(a, (2,2)).max((2,3))
    array([[4, 5],
           [7, 8]])
           
    Or delay embedding (3D embedding with delay 2):
    >>> x = np.arange(10)
    >>> rolling_window(x, 3, wsteps=2)
    array([[0, 2, 4],
           [1, 3, 5],
           [2, 4, 6],
           [3, 5, 7],
           [4, 6, 8],
           [5, 7, 9]])
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int) # maybe crude to cast to int...
    
    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w
    
    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.") 

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps):] = asteps
        
        if np.any(asteps < 1):
             raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps
    
    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
             raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1 # make sure that steps are 1 for non-existing dims.
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape # just renaming...
    
    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window==0] = 1
    
    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape
    
    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps
    
    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window):] = window
        _window = _.copy()
        _[-len(window):] = new_strides
        _new_strides = _
        
        new_shape = np.zeros(len(shape)*2, dtype=int)
        new_strides = np.zeros(len(shape)*2, dtype=int)
        
        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides
    
    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]
    
    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)

def apply_model_to_data(data, r, kmeans):
    height, width, _ = data.shape
    
    windowed_data = rolling_window(data, (2*r+1,2*r+1,0))
    windowed_data = np.rollaxis(windowed_data, 2, 5)
    windowed_data = windowed_data.reshape(int((height-2*r)*(width-2*r)), -1)
    
    labels = kmeans.predict(windowed_data)
    labels = labels.reshape(height-2*r, width-2*r)
    
    return labels


def image_cutout_builder(mask_size=(5,20), replacement_val=128):
    if isinstance(mask_size, tuple):
        size = np.random.randint(*mask_size)
    elif isinstance(mask_size, int):
        size = mask_size
    else:
        raise ValueError("mask_size must be int or tuple")

    def augment_function(image):
        height, width, _ = image.shape

        x = np.random.randint(0, width-size)
        y = np.random.randint(0, height-size)

        #mean_vals = np.mean(image, axis=(0,1))

        image[y:y+size, x:x+size] = replacement_val
        return image

    return augment_function


def basic_model(input_shape, num_classes, num_filters_per_layer=64, lr=0.003):
    inputs = Input(input_shape)

    x1 = Conv2D(num_filters_per_layer, kernel_size=(3,3), strides=(1,1), padding="same", activation=None)(inputs)
    x1 = LeakyReLU(0.1)(x1)
    x2 = Conv2D(num_filters_per_layer, kernel_size=(5,5), strides=(1,1), padding="same", activation=None)(inputs)
    x2 = LeakyReLU(0.1)(x2)
    x3 = Conv2D(num_filters_per_layer, kernel_size=(7,7), strides=(1,1), padding="same", activation=None)(inputs)
    x3 = LeakyReLU(0.1)(x3)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(num_filters_per_layer, kernel_size=(1,1), strides=(1,1), padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x1 = Conv2D(num_filters_per_layer, kernel_size=(3,3), strides=(1,1), padding="same", activation=None)(x)
    x1 = LeakyReLU(0.1)(x1)
    x2 = Conv2D(num_filters_per_layer, kernel_size=(5,5), strides=(1,1), padding="same", activation=None)(x)
    x2 = LeakyReLU(0.1)(x2)
    x3 = Conv2D(num_filters_per_layer, kernel_size=(7,7), strides=(1,1), padding="same", activation=None)(x)
    x3 = LeakyReLU(0.1)(x3)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(num_filters_per_layer, kernel_size=(1,1), strides=(1,1), padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x1 = Conv2D(num_filters_per_layer, kernel_size=(3,3), strides=(1,1), padding="same", activation=None)(x)
    x1 = LeakyReLU(0.1)(x1)
    x2 = Conv2D(num_filters_per_layer, kernel_size=(5,5), strides=(1,1), padding="same", activation=None)(x)
    x2 = LeakyReLU(0.1)(x2)
    x3 = Conv2D(num_filters_per_layer, kernel_size=(7,7), strides=(1,1), padding="same", activation=None)(x)
    x3 = LeakyReLU(0.1)(x3)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(num_filters_per_layer, kernel_size=(1,1), strides=(1,1), padding="same", activation=None)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    outputs = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), padding="same", activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(lr=lr)
    model.compile(loss=categorical_crossentropy, optimizer=optimizer)
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Unsupervised pre-training")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--input_fn", action="store", dest="input_fn", type=str, help="Path/paths to input GeoTIFF", required=True)
    parser.add_argument("--output_fn", action="store", dest="output_fn", type=str, help="Output model fn format (use '{epoch:02d}' for checkpointing per epoch of traning)", required=True)
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", required=True)

    args = parser.parse_args(sys.argv[1:])

    # Other args
    nodata_value = 0
    num_cluster_samples = 100000
    num_cluster_classes = 40
    r = 2
    num_train_samples = 5000
    num_val_samples = 50
    sample_height = 150
    sample_width = 150
    num_epochs = 30
    batch_size = 16
    assert sample_height % (2*r+1) == 0
    assert sample_width % (2*r+1) == 0

    #--------------------------------------------------
    print("Starting unsupervised pre-training script")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    start_time = float(time.time())

    #--------------------------------------------------
    print("Loading data")
    tic = float(time.time())
    all_data = []
    with rasterio.open(args.input_fn) as f:
        all_data = np.rollaxis(f.read(), 0, 3)
    height, width, num_channels = all_data.shape
    mask = np.sum(all_data == nodata_value, axis=2) == num_channels # We assume that if the `nodata_value` is present across all channels then the pixel is actually nodata
    print("Finished loading data in %0.4f seconds" % (time.time() - tic))


    #--------------------------------------------------
    # We first randomly sample (2*r+1, 2*r+1) patches to use to fit a KMeans model.
    # We will later sample (sample_height, sample_width) patches, then apply this KMeans
    # model to every pixel within those patches to get corresponding target labels.
    print("Sampling dataset for KMeans and fitting model")
    tic = float(time.time())
    x_all = np.zeros((num_cluster_samples, 2*r+1, 2*r+1, num_channels), dtype=float)
    for i in range(num_cluster_samples):

        x = np.random.randint(r, width-r)
        y = np.random.randint(r, height-r)
        while mask[y,x]: # we hit a no data
            x = np.random.randint(r, width-r)
            y = np.random.randint(r, height-r)

        x_window = all_data[y-r:y+r+1, x-r:x+r+1]
        x_all[i] = x_window
    x_all_flat = x_all.reshape((num_cluster_samples, -1))

    kmeans = KMeans(n_clusters=num_cluster_classes, verbose=0, n_init=20)
    kmeans = kmeans.fit(x_all_flat)
    print("Finished fitting KMeans in %0.4f seconds" % (time.time() - tic))


    #--------------------------------------------------
    print("Sampling training dataset")
    tic = float(time.time())

    x_train = np.zeros((num_train_samples,sample_height,sample_width,num_channels), dtype=float)
    x_val = np.zeros((num_val_samples,sample_height,sample_width,num_channels), dtype=float)

    y_train = np.zeros((num_train_samples,sample_height,sample_width), dtype=int)
    y_val = np.zeros((num_val_samples,sample_height,sample_width), dtype=int)

    for i in range(num_train_samples):
        if i % 1000 == 0:
            print("%d/%d" % (i, num_train_samples))
        x = np.random.randint(r, width-sample_width-r)
        y = np.random.randint(r, height-sample_height-r)
        
        while mask[y,x]:
            x = np.random.randint(r, width-sample_width-r)
            y = np.random.randint(r, height-sample_height-r)
        
        window = all_data[y-r:y+sample_height+r, x-r:x+sample_width+r]
        labels = apply_model_to_data(window, r, kmeans)
        x_train[i] = all_data[y:y+sample_height,x:x+sample_width].copy()
        y_train[i] = labels
    print("Finished sampling training dataset in %0.4f seconds" % (time.time() - tic))


    #--------------------------------------------------
    print("Sampling validation dataset")
    tic = float(time.time())
    for i in range(num_val_samples):
        if i % 1000 == 0:
            print("%d/%d" % (i, num_val_samples))
        x = np.random.randint(r, width-sample_width-r)
        y = np.random.randint(r, height-sample_height-r)
        
        while mask[y,x]:
            x = np.random.randint(r, width-sample_width-r)
            y = np.random.randint(r, height-sample_height-r)
        
        window = all_data[y-r:y+sample_height+r, x-r:x+sample_width+r]
        labels = apply_model_to_data(window, r, kmeans)
        x_val[i] = all_data[y:y+sample_height,x:x+sample_width].copy()
        y_val[i] = labels
    print("Finished sampling validation dataset in %0.4f seconds" % (time.time() - tic))
    

    #--------------------------------------------------
    print("Converting labels to categorical")
    tic = float(time.time())
    y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=num_cluster_classes)
    y_val = tensorflow.keras.utils.to_categorical(y_val, num_classes=num_cluster_classes)
    print("Finished converting labels to categorical in %0.4f seconds" % (time.time() - tic))

    
    #--------------------------------------------------
    print("Creating and fitting model")
    tic = float(time.time())
    model = basic_model((sample_height, sample_width, num_channels), num_cluster_classes, lr=0.01)
    if args.verbose:
        model.summary()

    datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        channel_shift_range=0.1,
        fill_mode='constant', # can also be "nearest"
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        preprocessing_function=image_cutout_builder(mask_size=(5,20), replacement_val=128),
        dtype=np.float32
    )

    period = x_train.shape[0] // batch_size - 1
    model_checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(args.output_fn, monitor='train_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', save_freq=period)
    lr_reducer = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2, verbose=1)

    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=x_train.shape[0] // batch_size - 1,
        epochs=num_epochs,
        callbacks=[model_checkpoint, lr_reducer],
        validation_data=datagen.flow(x_val, y_val, batch_size=batch_size),
        validation_steps=x_val.shape[0] // batch_size - 1
    )
    print("Finished fitting model in %0.4f seconds" % (time.time() - tic))

    print("Finished in %0.4f seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
