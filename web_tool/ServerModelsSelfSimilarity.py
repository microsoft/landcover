import sys
import os
import time
import numpy as np

import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scipy.signal

from .ServerModelsAbstract import BackendModel
from .Utils import to_categorical

def softmax(x, theta = 1.0, axis = -1):
    x = (x*theta)
    exp_max = np.exp(x - np.max(x,axis=axis,keepdims=True))
    out = exp_max/np.sum(exp_max,axis=axis,keepdims=True)
    return out

class BasicFineTune(BackendModel):

    def __init__(self, model_fn, verbose=False):

        self.model_fn = model_fn
        self.model = joblib.load(self.model_fn)
        self.K = 3
        assert self.K % 2 == 1

        self.augment_x_train = []
        self.augment_y_train = []
        self.undo_stack = []

        self.current_features = None

        print("Model created")
     
    def run(self, naip_data, extent, on_tile=False):
        ''' Expects naip_data to have shape (height, width, channels) and have values in the [0, 255] range.
        '''
        naip_data = naip_data / 255.0
        output, output_features = self.run_model_on_tile(naip_data)
        
        if not on_tile:
            self.current_features = output_features

        return output

    def retrain(self, **kwargs):
        return True, ""
        
    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        x_features = self.current_features[tdst_row:bdst_row+1, tdst_col:bdst_col+1, :].copy().reshape(-1, self.current_features.shape[2])

        y_samples = np.zeros((x_features.shape[0]), dtype=np.uint8)
        y_samples[:] = class_idx

        #x_features_transformed = 3 * np.log(x_features + 1e-4)

        self.augment_x_train.append(x_features)
        self.augment_y_train.append(y_samples)
        self.undo_stack.append("sample")

    def undo(self):
        num_undone = 0
        if len(self.undo_stack) > 0:
            undo = self.undo_stack.pop()
            if undo == "sample":
                self.augment_x_train.pop()
                self.augment_y_train.pop()
                num_undone += 1
                success = True
                message = "Undoing sample"
            elif undo == "retrain":
                while self.undo_stack[-1] == "retrain":
                    self.undo_stack.pop()
                self.augment_x_train.pop()
                self.augment_y_train.pop()
                num_undone += 1
                success = True
                message = "Undoing sample"
            else:
                raise ValueError("This shouldn't happen")
        else:
            success = False
            message = "Nothing to undo"
        return success, message, num_undone

    def reset(self):
        self.augment_x_train = []
        self.augment_y_train = []
        self.undo_stack = []
        self.retrain()

    def run_model_on_tile(self, naip_tile):
        height, width, num_channels = naip_tile.shape
        naip_tile = naip_tile.reshape(-1, num_channels)

        K = self.K
        q = self.model.predict_proba(naip_tile)
        
        _, num_clusters = q.shape

        features = np.zeros((height, width, num_clusters))

        q = q.reshape(height, width, num_clusters)
        # Q = np.ones((height-K, width-K, num_clusters))
        # tmp = np.zeros((height, width))

        # for i in range(num_channels):

        #     tmp[:] = q[:,:,i].copy()
        #     cm = np.cumsum(np.cumsum(tmp, axis=1), axis=0)
        #     Q[:,:,i] = cm[K:,K:] + cm[:-K,:-K] - cm[K:,:-K] - cm[:-K, K:]
        # Q = Q / Q.sum(axis=2, keepdims=True)
        # features[self.pad_start:-self.pad_end, self.pad_start:-self.pad_end] = Q.copy()


        kernel = np.ones((K,K))
        for i in range(num_channels):
            features[:,:,i] = scipy.signal.correlate2d(q[:,:,i], kernel, mode="same")


        if len(self.augment_x_train) == 0:      
            return np.zeros((height, width, num_channels)), features 
        else:
            # actually classify these
            x_train = np.concatenate(self.augment_x_train, axis=0)
            y_train = np.concatenate(self.augment_y_train, axis=0)
            y_train = to_categorical(y_train)
            _, num_classes = y_train.shape

            y_pred = np.zeros((height, width, num_classes))

            Q = features.copy()
            Q = Q.reshape(-1, num_clusters)
            Q = 5.0 * np.log(Q + 1e-8)

            lP = Q @ x_train.T
            lP = np.exp(lP) @ y_train

            lP = lP / np.sum(lP, axis=1, keepdims=True)
            
            print(lP.shape, lP.min(), lP.max())

            y_pred = lP.reshape(height, width, num_classes)
            #y_pred = np.zeros((height,width,y_train.shape[1]))
            #y_pred[self.pad_start:-self.pad_end, self.pad_start:-self.pad_end] = lP.reshape(height-K, width-K, num_classes)
            return y_pred, features
