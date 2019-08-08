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

import keras.backend as K
import keras.callbacks
import keras.utils
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.models import Model
from keras.layers import Input, Dense, Activation, MaxPooling2D, Conv2D, BatchNormalization
from keras.layers import Concatenate, Cropping2D, Lambda
from keras.losses import categorical_crossentropy, mean_squared_error

from sklearn.cluster import MiniBatchKMeans

def basic_model(input_shape, num_classes, lr=0.003):
    inputs = Input(input_shape)

    inputs = Input(input_shape)

    x1 = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(inputs)
    x2 = Conv2D(128, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(inputs)
    x3 = Conv2D(128, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(inputs)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x1 = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(x)
    x2 = Conv2D(128, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(x)
    x3 = Conv2D(128, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(x)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)

    x1 = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu")(x)
    x2 = Conv2D(128, kernel_size=(5,5), strides=(1,1), padding="same", activation="relu")(x)
    x3 = Conv2D(128, kernel_size=(7,7), strides=(1,1), padding="same", activation="relu")(x)
    x = Concatenate(axis=-1)([x1,x2,x3])
    x = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding="same", activation="relu")(x)

    outputs = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), padding="same", activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(lr=lr)
    model.compile(loss=categorical_crossentropy, optimizer=optimizer)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Autoencoder training")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--input_fn", action="store", dest="input_fn", type=str, help="Path to input GeoTIFF", required=True)
    parser.add_argument("--output_fn", action="store", dest="output_fn", type=str, help="Output model fn format", required=True)
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", required=True)

    args = parser.parse_args(sys.argv[1:])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    start_time = float(time.time())
    
    
    print("Loading data")
    f = rasterio.open(args.input_fn,"r")
    data = np.rollaxis(f.read(), 0, 3)
    f.close()

    data = np.concatenate([
        data,
        data[:,:,0][:,:,np.newaxis]
    ], axis=2)

    assert data.shape[2] == 4
    print("Loaded data with shape:", data.shape)

    print("Fitting KMeans model for labels")
    num_classes = 10
    data_original_shape = data.shape
    data_color_features = data.reshape(-1,4)
    kmeans = MiniBatchKMeans(n_clusters=num_classes, verbose=1)
    labels = kmeans.fit_predict(data_color_features)
    data_color_labels = labels.reshape(data_original_shape[:2])

    print("Extracting training samples")
    n_samples = 4000
    height, width = 100, 100
    x_all = np.zeros((n_samples, height, width, 4), dtype=np.float32)
    y_all = np.zeros((n_samples, height, width), dtype=np.float32)

    for i in range(n_samples):
        x = np.random.randint(0, data.shape[1]-width)
        y = np.random.randint(0, data.shape[0]-height)
        
        img = data[y:y+height, x:x+width, :].astype(np.float32)
        target = data_color_labels[y:y+height, x:x+width].copy()
        
        x_all[i] = img
        y_all[i] = target

    x_all = x_all/255.0
    y_all = keras.utils.to_categorical(y_all)


    model = basic_model((height, width, 4), num_classes, lr=0.003)
    model.summary()

    print("Fitting model")
    model_checkpoint = keras.callbacks.ModelCheckpoint(args.output_fn, monitor='train_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    model.fit(x_all, y_all, batch_size=32, epochs=30, verbose=1, callbacks=[model_checkpoint], validation_split=0.1)


    print("Finished in %0.4f seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
