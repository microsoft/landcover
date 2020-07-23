import sys, os, time, copy
import numpy as np

import sklearn.base
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
import tensorflow.keras as keras

import logging
LOGGER = logging.getLogger("server")

from . import ROOT_DIR
from .ModelSessionAbstract import ModelSession

import scipy.optimize

def softmax(z):
    assert len(z.shape) == 2
    shift = np.max(z, axis=1, keepdims=True)
    e_z = np.exp(z - shift)
    denominator = np.sum(e_z, axis=1, keepdims=True)
    return e_z / denominator

def pred(X, W, b):
    return softmax(X@W + b)

def nll(X, W, b, y):
    if len(X.shape) == 1:
        X = X[np.newaxis,:]
    y_pred = pred(X, W, b)  
    loss = (-np.log(y_pred)) * y #y is one hot encoded, so we zero out everything except for the "correct" class
    return np.sum(loss)


class KerasDenseFineTune(ModelSession):

    # AUGMENT_MODEL = MLPClassifier(
    #     hidden_layer_sizes=(),
    #     activation='relu',
    #     alpha=0.0001,
    #     solver='lbfgs',
    #     tol=0.0001,
    #     verbose=False,
    #     validation_fraction=0.0,
    #     n_iter_no_change=50
    # )

    AUGMENT_MODEL = RandomForestClassifier()

    def __init__(self, model_fn, gpuid, fine_tune_layer, verbose=False):

        self.model_fn = model_fn
        

        tmodel = keras.models.load_model(self.model_fn, compile=False, custom_objects={
            "jaccard_loss":keras.metrics.mean_squared_error, 
            "loss":keras.metrics.mean_squared_error
        })

        feature_layer_idx = fine_tune_layer
        tmodel.summary()
        
        self.model = keras.models.Model(inputs=tmodel.inputs, outputs=[tmodel.outputs[0], tmodel.layers[feature_layer_idx].output])
        self.model.compile("sgd","mse")
        #self.model._make_predict_function()	# have to initialize before threading

        self.output_channels = self.model.output_shape[0][3]
        self.output_features = self.model.output_shape[1][3]
        self.input_size = self.model.input_shape[1]

        self.down_weight_padding = 10

        self.stride_x = self.input_size - self.down_weight_padding*2
        self.stride_y = self.input_size - self.down_weight_padding*2

        self.verbose = verbose

        # Seed augmentation model dataset
        self.current_features = None

        self.augment_base_x_train = []
        self.augment_base_y_train = []

        self.augment_x_train = []
        self.augment_y_train = []
        self.augment_model = sklearn.base.clone(KerasDenseFineTune.AUGMENT_MODEL)
        self.augment_model_trained = False

        self.undo_stack = []

        ## Setting up "seed" data
        self.use_seed_data = False


        '''
        # Find the weights of the last convolutional layer, assume this is 1x1 convs (or a dense layer) with bias
        for layer_idx in range(-1,-5,-1):
            weights = tmodel.layers[layer_idx].get_weights()
            if len(weights) == 2:
                W = weights[0]
                W = W.squeeze()
                b = weights[1]
                break

        num_features, num_classes = W.shape
        for y_idx in range(1,num_classes):
            X = np.random.random((num_features))

            y = np.zeros((1,num_classes), dtype=np.float32)
            y[0,y_idx] = 1

            bounds = [
                (-1,1)
                for i in range(num_features)
            ]

            result = scipy.optimize.minimize(nll, X, args=(W,b,y), options={"maxiter":5}, bounds=bounds)

            X = result.x[np.newaxis,:]

            print("Class %d\tObjective %0.2f, Prediction %s" % (
                y_idx, nll(X,W,b,y), str(pred(X,W,b).round(2))
            ))

            self.augment_base_x_train.append(X)
            self.augment_base_y_train.append([y_idx-1])

        # Add seed data to normal training data
        for row in self.augment_base_x_train:
            self.augment_x_train.append(row)
        for row in self.augment_base_y_train:
            self.augment_y_train.append(row)
        '''
     
    @property
    def last_tile(self):
        return 0

    def run(self, tile, inference_mode=False):
        ''' Expects tile to have shape (height, width, channels) and have values in the [0, 255] range.
        '''
        tile = tile / 255.0
        output, output_features = self.run_model_on_tile(tile)
        
        if self.augment_model_trained:
            original_shape = output.shape
            output = output_features.reshape(-1, output_features.shape[2])
            output = self.augment_model.predict_proba(output)
            output = output.reshape(original_shape[0], original_shape[1],  -1)

        if not inference_mode:
            self.current_features = output_features

        return output

    def run_model_on_batch(self, batch_data, batch_size=32, predict_central_pixel_only=False):
        ''' Expects batch_data to have shape (none, 240, 240, 4) and have values in the [0, 255] range.
        '''
        batch_data = batch_data / 255.0
        output = self.model.predict(batch_data, batch_size=batch_size, verbose=0)
        output, output_features = output
        output = output[:,:,:,1:]

        if self.augment_model_trained:
            num_samples, height, width, num_features = output_features.shape

            if predict_central_pixel_only:
                output_features = output_features[:,120,120,:].reshape(-1, num_features)
                output = self.augment_model.predict_proba(output_features)
                output = output.reshape(num_samples, 4)
            else:
                output_features = output_features.reshape(-1, num_features)
                output = self.augment_model.predict_proba(output_features)
                output = output.reshape(num_samples, height, width, 4)
        else:
            if predict_central_pixel_only:
                output = output[:,120,120,:]
        
        return output

    def retrain(self, **kwargs):
        x_train = np.array(self.augment_x_train)
        y_train = np.array(self.augment_y_train)

        print(x_train.shape)
        print(y_train.shape)
        
        vals, counts = np.unique(y_train, return_counts=True)

        if len(vals) >= 4:
            self.augment_model.fit(x_train, y_train)
            LOGGER.debug("Fine-tuning accuracy: %0.4f" % (self.augment_model.score(x_train, y_train)))
            self.augment_model_trained = True
            self.undo_stack.append("retrain")

            success = True
            message = "Fit accessory model with %d samples" % (x_train.shape[0])
        else:
            success = False
            message = "Need to include training samples from each class"
        
        return success, message
        
    def add_sample_point(self, row, col, class_idx):
        self.augment_x_train.append(self.current_features[row, col, :].copy())
        self.augment_y_train.append(class_idx)
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
        self.augment_model = sklearn.base.clone(KerasDenseFineTune.AUGMENT_MODEL)
        self.augment_model_trained = False

        for row in self.augment_base_x_train:
            self.augment_x_train.append(row)
        for row in self.augment_base_y_train:
            self.augment_y_train.append(row)

        if self.use_seed_data:
            self.retrain()

    def run_model_on_tile(self, naip_tile, batch_size=32):
        ''' Expects naip_tile to have shape (height, width, channels) and have values in the [0, 1] range.
        '''
        height = naip_tile.shape[0]
        width = naip_tile.shape[1]
        
        output = np.zeros((height, width, self.output_channels), dtype=np.float32)
        output_features = np.zeros((height, width, self.output_features), dtype=np.float32)

        counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
        kernel = np.ones((self.input_size, self.input_size), dtype=np.float32) * 0.1
        kernel[10:-10, 10:-10] = 1
        kernel[self.down_weight_padding:self.down_weight_padding+self.stride_y,
               self.down_weight_padding:self.down_weight_padding+self.stride_x] = 5

        batch = []
        batch_indices = []
        batch_count = 0

        for y_index in (list(range(0, height - self.input_size, self.stride_y)) + [height - self.input_size,]):
            for x_index in (list(range(0, width - self.input_size, self.stride_x)) + [width - self.input_size,]):
                naip_im = naip_tile[y_index:y_index+self.input_size, x_index:x_index+self.input_size, :]

                batch.append(naip_im)
                batch_indices.append((y_index, x_index))
                batch_count+=1


        model_output = self.model.predict(np.array(batch), batch_size=batch_size, verbose=0)
        
        for i, (y, x) in enumerate(batch_indices):
            output[y:y+self.input_size, x:x+self.input_size] += model_output[0][i] * kernel[..., np.newaxis]
            output_features[y:y+self.input_size, x:x+self.input_size] += model_output[1][i] * kernel[..., np.newaxis]
            counts[y:y+self.input_size, x:x+self.input_size] += kernel

        output = output / counts[..., np.newaxis]
        output = output[:,:,1:]
        output_features = output_features / counts[..., np.newaxis]

        return output, output_features


    def save_state_to(self, directory):
        pass

    def load_state_from(self, directory):
        pass