import sys
sys.path.append("..")
import os
import time
import copy
import json

import logging
LOGGER = logging.getLogger("server")

import numpy as np

import sklearn.base
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelBinarizer

from .ModelSessionAbstract import ModelSession
from training.models.unet import Unet

import torch
import torch.nn as nn

def softmax(output):
    output_max = np.max(output, axis=2, keepdims=True)
    exps = np.exp(output-output_max)
    exp_sums = np.sum(exps, axis=2, keepdims=True)
    return exps/exp_sums    
    
class TorchFineTuning(ModelSession):

    AUGMENT_MODEL = SGDClassifier(
        loss="log",
        shuffle=True,
        n_jobs=-1,
        learning_rate="constant",
        eta0=0.001,
        warm_start=True
    )

    def __init__(self, model_fn, gpu_id, fine_tune_layer):
        self.model_fn = model_fn
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.output_channels = 14
        self.output_features = 64
        self.input_size = 128

        self.down_weight_padding = 20

        self.stride_x = self.input_size - self.down_weight_padding*2
        self.stride_y = self.input_size - self.down_weight_padding*2

        self.model = Unet(
            feature_scale=1,
            n_classes=14,
            in_channels=6,
            is_deconv=True,
            is_batchnorm=True
        )

        self._init_model()
        for param in self.model.parameters():
            param.requires_grad = False

        self.initial_weights = self.model.final.weight.cpu().detach().numpy().squeeze()
        self.initial_biases = self.model.final.bias.cpu().detach().numpy()

        self.augment_model = sklearn.base.clone(TorchFineTuning.AUGMENT_MODEL)
        self.augment_model_trained = False

        self.augment_model.coef_ = self.initial_weights.astype(np.float64)
        self.augment_model.intercept_ = self.initial_biases.astype(np.float64)

        self.augment_model.classes_ = np.array(list(range(self.output_channels)))
        self.augment_model.n_features_in_ = self.output_features


        self.img_data = None
        self.current_features = None
        self.current_output = None
        
        self.corr_features = []
        self.corr_labels = []

    def _init_model(self):
        checkpoint = torch.load(self.model_fn, map_location=self.device)["state_dict"]
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model = self.model.to(self.device)


    def run(self, img_data, naip_fn, extent):
        print("In run()", img_data.shape)
        
        x = img_data.copy()
        output, output_features = self.run_model_on_tile(x)

        self.img_data = img_data
        self.current_features = output_features
        self.current_output = output

        return output

    def retrain(self, **kwargs):
        x_train = np.array(self.corr_features)
        y_train = np.array(self.corr_labels)
        
        
        self.augment_model.partial_fit(x_train, y_train)
        LOGGER.debug("Fine-tuning accuracy: %0.4f" % (self.augment_model.score(x_train, y_train)))
            

        new_weights = torch.from_numpy(self.augment_model.coef_.copy().astype(np.float32)[:,:,np.newaxis,np.newaxis])
        new_biases = torch.from_numpy(self.augment_model.intercept_.astype(np.float32))
        new_weights = new_weights.to(self.device)
        new_biases = new_biases.to(self.device)

        print(new_weights.shape)
        print(new_biases.shape)

        self.model.final.weight.data = new_weights
        self.model.final.bias.data = new_biases

        success = True
        message = "Fit last layer model with %d samples" % (x_train.shape[0])
        
        return success, message

    # def retrain(self, train_steps=100, learning_rate=1e-3):
      
    #     print_every_k_steps = 10

    #     print("Fine tuning with %d new labels." % self.num_corrected_pixels)
    #     batch_x = torch.from_numpy(np.array(self.corr_features)).float().to(self.device)
    #     batch_y = torch.from_numpy(np.array(self.corr_labels)).to(self.device)
        
    #     self._init_model()
        
    #     optimizer = torch.optim.Adam(self.model.last.parameters(), lr=learning_rate, eps=1e-5)
        
    #     criterion = nn.CrossEntropyLoss().to(self.device)

    #     for i in range(train_steps):
    #         #print('step %d' % i)
    #         acc = 0
            
    #         with torch.enable_grad():

    #             optimizer.zero_grad()
                
    #             pred = self.model.last.forward(batch_x.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
    #             #print(pred)
    #             #print('retr', y_pred1.shape, out.shape)

    #             loss = criterion(pred,batch_y)
                
    #             print(loss.mean().item())
                
    #             acc = (pred.argmax(1)==batch_y).float().mean().item()

    #             loss.backward()
    #             optimizer.step()
            
    #         if i % print_every_k_steps == 0:
    #             print("Step pixel acc: ", acc)

    #     success = True
    #     message = "Fine-tuned model with %d samples." % len(self.corr_features)
    #     print(message)
    #     return success, message
    
    def undo(self):
        if len(self.corr_features)>0:
            self.corr_features = self.corr_features[:-1]
            self.corr_labels = self.corr_labels[:-1]

    def add_sample_point(self, row, col, class_idx):
        self.corr_labels.append(class_idx)
        self.corr_features.append(self.current_features[row, col,:])
        
    def reset(self):
        self._init_model()
        self.corr_features = []
        self.corr_labels = []
        self.augment_model = sklearn.base.clone(TorchFineTuning.AUGMENT_MODEL)
        self.augment_model_trained = False

        label_binarizer = LabelBinarizer()
        label_binarizer.fit(range(self.output_channels))

        self.augment_model.coefs_ = [self.initial_weights]
        self.augment_model.intercepts_ = [self.initial_biases]

        self.augment_model.classes_ = np.array(list(range(self.output_channels)))
        self.augment_model.n_features_in_ = self.output_features
        self.augment_model.n_outputs_ = self.output_channels
        self.augment_model.n_layers_ = 2
        self.augment_model.out_activation_ = 'softmax'

        self.augment_model._label_binarizer = label_binarizer


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
        batch = np.array(batch)

        model_output = []
        model_feature_output = []
        for i in range(0, len(batch), batch_size):

            t_batch = batch[i:i+batch_size]
            t_batch = np.rollaxis(t_batch, 3, 1)
            t_batch = torch.from_numpy(t_batch).to(self.device)
            
            with torch.no_grad():
                predictions, features = self.model.forward_features(t_batch)
                predictions = predictions.cpu().numpy()
                features = features.cpu().numpy()

            print(predictions.shape, features.shape)
            predictions = np.rollaxis(predictions, 1, 4)
            features = np.rollaxis(features, 1, 4)

            model_output.append(predictions)
            model_feature_output.append(features)

        model_output = np.concatenate(model_output, axis=0)
        model_feature_output = np.concatenate(model_feature_output, axis=0)
        
        for i, (y, x) in enumerate(batch_indices):
            output[y:y+self.input_size, x:x+self.input_size] += model_output[i] * kernel[..., np.newaxis]
            output_features[y:y+self.input_size, x:x+self.input_size] += model_feature_output[i] * kernel[..., np.newaxis]
            counts[y:y+self.input_size, x:x+self.input_size] += kernel

        output = output / counts[..., np.newaxis]
        #output = output[:,:,1:]
        output_features = output_features / counts[..., np.newaxis]

        return softmax(output), output_features