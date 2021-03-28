#! /usr/bin/env python
import os
import time
import glob
import pickle
import argparse

import numpy as np
import rasterio

from sklearn.cluster import KMeans


parser = argparse.ArgumentParser(description="Unsupervised model training")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
parser.add_argument("--input_fn", action="store", type=str, help="Path/paths to input GeoTIFF", required=True)
parser.add_argument("--nodata_val", action="store", type=int, default=0, help="Value to treat as nodata in the input imagery")
parser.add_argument("--output_fn", action="store", type=str, help="Output model file (the best model according to 'val_loss' will be saved here)", required=True)

# Cluster step arguments
parser.add_argument("--num_clusters", type=int, default=64, help="Number of clusters to use in the k-means model")
parser.add_argument("--num_cluster_samples_per_file", type=int, default=1000, help="Number of pixels (or neighborhoods) to sample and use to fit the k-means model")
parser.add_argument("--radius", type=int, default=2, help="Size of neighborhood to use in creating the samples with the k-means model")

# Training step arguments
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training model")
parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs to train model for")
parser.add_argument("--patch_size", type=int, default=150, help="The size of each patch to sample in order to train the DL model")
parser.add_argument("--num_train_samples_per_file", type=int, default=1000, help="Number of patches to sample and use to fit the DL model")
parser.add_argument("--num_val_samples_per_file", type=int, default=50, help="Number of patches to sample and use as validation")

parser.add_argument("--normalization_means", type=str, required=False, help="Comma separated string of the per-channel means to subtract before training the model")
parser.add_argument("--normalization_stds", type=str, required=False, help="Comma separated string of the per-channel stdevs to divide by before training the model")

parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", required=True)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
print("Using tensorflow version: %s" % (tf.__version__))
print("Is GPU available: ", tf.test.is_gpu_available())

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
    model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])

    return model

def main():
    # Other args
    nodata_value = args.nodata_val
    num_cluster_samples_per_file = args.num_cluster_samples_per_file
    num_clusters = args.num_clusters
    r = args.radius
    num_train_samples_per_file = args.num_train_samples_per_file
    num_val_samples_per_file = args.num_val_samples_per_file
    sample_size = args.patch_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    initial_lr = 0.01
    assert sample_size % (2*r+1) == 0
    assert sample_size % (2*r+1) == 0

    input_fns = glob.glob(args.input_fn)
    num_files = len(input_fns)
    num_cluster_samples = num_cluster_samples_per_file * num_files
    num_train_samples = num_train_samples_per_file * num_files
    num_val_samples = num_val_samples_per_file * num_files
    output_dir = os.path.dirname(args.output_fn)

    #--------------------------------------------------
    print("Starting unsupervised pre-training script with %d inputs" % (len(input_fns)))
    start_time = float(time.time())

    #--------------------------------------------------
    print("Loading data")
    tic = float(time.time())

    all_data = []
    for fn in input_fns:
        with rasterio.open(fn) as f:
            all_data.append(np.rollaxis(f.read(), 0, 3))

    _, __, num_channels = all_data[0].shape
    all_masks = []
    for data in all_data:
        assert data.shape[2] == num_channels
        all_masks.append(np.sum(data == nodata_value, axis=2) == num_channels) # We assume that if the `nodata_value` is present across all channels then the pixel is actually nodata
    print("Finished loading %d files in %0.4f seconds" % (len(all_data), time.time() - tic))


    #--------------------------------------------------
    # We first randomly sample (2*r+1, 2*r+1) patches to use to fit a KMeans model.
    # We will later sample (sample_size, sample_size) patches, then apply this KMeans
    # model to every pixel within those patches to get corresponding target labels.
    print("Sampling dataset for KMeans and fitting model")
    tic = float(time.time())
    x_all = np.zeros((num_cluster_samples, 2*r+1, 2*r+1, num_channels), dtype=float)
    idx = 0
    for data, mask in zip(all_data, all_masks):
        height, width, _ = data.shape
        for i in range(num_cluster_samples_per_file):

            x = np.random.randint(r, width-r)
            y = np.random.randint(r, height-r)
            while mask[y,x]: # we hit a no data
                x = np.random.randint(r, width-r)
                y = np.random.randint(r, height-r)

            x_all[idx] = data[y-r:y+r+1, x-r:x+r+1].copy()
            idx += 1
    x_all_flat = x_all.reshape((num_cluster_samples, -1))

    kmeans = KMeans(n_clusters=num_clusters, verbose=0, n_init=20)
    kmeans = kmeans.fit(x_all_flat)
    print("Finished fitting KMeans in %0.4f seconds" % (time.time() - tic))


    #--------------------------------------------------
    print("Sampling training dataset")
    tic = float(time.time())

    x_train = np.zeros((num_train_samples,sample_size,sample_size,num_channels), dtype=float)
    x_val = np.zeros((num_val_samples,sample_size,sample_size,num_channels), dtype=float)

    y_train = np.zeros((num_train_samples,sample_size,sample_size), dtype=int)
    y_val = np.zeros((num_val_samples,sample_size,sample_size), dtype=int)
    idx = 0
    for data, mask in zip(all_data, all_masks):
        height, width, _ = data.shape
        for i in range(num_train_samples_per_file):
            if idx % 1000 == 0:
                print("%d/%d" % (idx, num_train_samples))
            x = np.random.randint(r, width-sample_size-r)
            y = np.random.randint(r, height-sample_size-r)

            while mask[y,x]:
                x = np.random.randint(r, width-sample_size-r)
                y = np.random.randint(r, height-sample_size-r)

            window = data[y-r:y+sample_size+r, x-r:x+sample_size+r]
            labels = apply_model_to_data(window, r, kmeans)
            x_train[idx] = data[y:y+sample_size,x:x+sample_size].copy()
            y_train[idx] = labels
            idx += 1
    print("Finished sampling training dataset in %0.4f seconds" % (time.time() - tic))


    #--------------------------------------------------
    print("Sampling validation dataset")
    tic = float(time.time())
    idx = 0
    for data, mask in zip(all_data, all_masks):
        height, width, _ = data.shape
        for i in range(num_val_samples_per_file):
            if idx % 1000 == 0:
                print("%d/%d" % (idx, num_val_samples))
            x = np.random.randint(r, width-sample_size-r)
            y = np.random.randint(r, height-sample_size-r)

            while mask[y,x]:
                x = np.random.randint(r, width-sample_size-r)
                y = np.random.randint(r, height-sample_size-r)

            window = data[y-r:y+sample_size+r, x-r:x+sample_size+r]
            labels = apply_model_to_data(window, r, kmeans)
            x_val[idx] = data[y:y+sample_size,x:x+sample_size].copy()
            y_val[idx] = labels
            idx += 1
    print("Finished sampling validation dataset in %0.4f seconds" % (time.time() - tic))


    #--------------------------------------------------
    print("Normalizing sampled imagery")
    means = 0
    stds = 1
    if (args.normalization_means is not None) and (args.normalization_stds is not None):
        means = np.array(list(map(float,args.normalization_means.split(","))))
        stds = np.array(list(map(float,args.normalization_stds.split(","))))
        assert means.shape[0] == 0 or means.shape[0] == num_channels
        assert stds.shape[0] == 0 or stds.shape[0] == num_channels
    x_train = (x_train - means) / stds
    x_val = (x_val - means) / stds


    #--------------------------------------------------
    print("Converting labels to categorical")
    tic = float(time.time())
    y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=num_clusters)
    y_val = tensorflow.keras.utils.to_categorical(y_val, num_classes=num_clusters)
    print("Finished converting labels to categorical in %0.4f seconds" % (time.time() - tic))

    #--------------------------------------------------
    print("Creating and fitting model")
    tic = float(time.time())
    model = basic_model((sample_size, sample_size, num_channels), num_clusters, lr=initial_lr)
    if args.verbose:
        model.summary()

    train_datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        channel_shift_range=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        preprocessing_function=image_cutout_builder(mask_size=(5,20), replacement_val=means),
        dtype=np.float32
    )
    val_datagen = ImageDataGenerator()

    model_checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(args.output_fn, monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=False, mode="min", save_freq="epoch")
    lr_reducer = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=2, verbose=1)
    early_stopper = tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=8)

    history = model.fit(
        train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
        steps_per_epoch=x_train.shape[0] // batch_size - 1,
        epochs=num_epochs,
        callbacks=[model_checkpoint, lr_reducer, early_stopper],
        validation_data=val_datagen.flow(x_val, y_val, batch_size=batch_size),
        validation_steps=x_val.shape[0] // batch_size - 1
    )

    print("Finished fitting model in %0.4f seconds" % (time.time() - tic))

    with open(os.path.join(output_dir, "model_fit_history.p"), "wb") as f:
        pickle.dump(
            history.history,
            f
        )

    print("Finished in %0.4f seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
