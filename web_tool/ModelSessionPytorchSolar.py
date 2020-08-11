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
from training.models.unet_solar import UnetModel

import torch
import torch.nn as nn
import pickle

def softmax(output):
    output_max = np.max(output, axis=2, keepdims=True)
    exps = np.exp(output-output_max)
    exp_sums = np.sum(exps, axis=2, keepdims=True)
    return exps/exp_sums 

def load_options(file_name):
    opt = pickle.load(open(file_name + '.pkl', 'rb'))
    return opt

class SolarFineTuning(ModelSession):

    AUGMENT_MODEL = SGDClassifier(
        loss="log",
        shuffle=True,
        n_jobs=-1,
        learning_rate="constant",
        eta0=0.001,
        warm_start=True
    )

    def __init__(self, gpuid, **kwargs):
        self.model_fn = os.path.join(kwargs["fn"] ,"training/checkpoint.pth.tar")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.opts = load_options(kwargs["fn"] +'/opt')

        self.output_channels = 2
        self.output_features = 16
        self.input_size = 256

        self.down_weight_padding = 20

        self.stride_x = self.input_size - self.down_weight_padding*2
        self.stride_y = self.input_size - self.down_weight_padding*2

        self.model = UnetModel(self.opts)

        self._init_model()
        for param in self.model.seg_layer.parameters():
            param.requires_grad = True

        self.initial_weights = self.model.seg_layer.weight.cpu().detach().numpy().squeeze()
        self.initial_biases = self.model.seg_layer.bias.cpu().detach().numpy()

        self.augment_model = sklearn.base.clone(SolarFineTuning.AUGMENT_MODEL)
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
        checkpoint = torch.load(self.model_fn, map_location=self.device)["model"]
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model = self.model.to(self.device)


    def run(self, img_data, inference_mode=False):
        print("In run()", img_data.shape)
        
        x = np.clip( (img_data.copy() / 3000.0), 0, 1)
        mean=[0.24314979229958464, 0.29540161742823484, 0.35506349200198395, 0.42654589817512967, 0.4846353036456073, 0.5264263955194577, 0.5668136121272241, 0.5630633021266042, 0.5982257347014694, 0.5799612925475852, 0.6510998369913404, 0.5493496367826067],
        std=[0.03478861968138082, 0.057298860977016156, 0.07412042681220939, 0.0986550527401778, 0.10487961908030127, 0.11762218219809227, 0.12738022812438296, 0.13463675723926963, 0.1354791923600473, 0.1162830350762486, 0.14922624332069723, 0.1278850911445811]
        x = (x-mean)/ std

        output, output_features = self.run_model_on_tile(x)

        self.img_data = img_data
        self.current_features = output_features
        self.current_output = output

        return output

    # def retrain(self, **kwargs):
    #     x_train = np.array(self.corr_features)
    #     y_train = np.array(self.corr_labels)
    #     print(x_train.shape)
    #     print(y_train.shape)
        
    #     self.augment_model.partial_fit(x_train, y_train)
    #    # LOGGER.debug("Fine-tuning accuracy: %0.4f" % (self.augment_model.score(x_train, y_train)))
            

    #     new_weights = torch.from_numpy(self.augment_model.coef_.copy().astype(np.float32)[:,:,np.newaxis,np.newaxis])
    #     new_biases = torch.from_numpy(self.augment_model.intercept_.astype(np.float32))
    #     new_weights = new_weights.to(self.device)
    #     new_biases = new_biases.to(self.device)


    #     self.model.seg_layer.weight.data = new_weights
    #     self.model.seg_layer.bias.data = new_biases

    #     success = True
    #     message = "Fit last layer model with %d samples" % (x_train.shape[0])
        
    #     return success, message

    def retrain(self, train_steps=100, learning_rate=1e-3):
      
        print_every_k_steps = 10

        batch_x = torch.from_numpy(np.array(self.corr_features)).float().to(self.device)
        batch_y = torch.from_numpy(np.array(self.corr_labels)).to(self.device)
        
        self._init_model()
        
        optimizer = torch.optim.Adam(self.model.seg_layer.parameters(), lr=learning_rate, eps=1e-5)
        
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for i in range(train_steps):
            #print('step %d' % i)
            acc = 0
            
            with torch.enable_grad():

                optimizer.zero_grad()
                
                pred = self.model.seg_layer.forward(batch_x.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
                
                loss = criterion(pred,batch_y)
                
                print(loss.mean().item())
                
                acc = (pred.argmax(1)==batch_y).float().mean().item()

                loss.backward()
                optimizer.step()
            
            if i % print_every_k_steps == 0:
                print("Step pixel acc: ", acc)

        return {
                "message": "Fine-tuned model with %d samples." % len(self.corr_features),
                "success": True
        }
    

    
    def undo(self):
        if len(self.corr_features)>0:
            self.corr_features = self.corr_features[:-1]
            self.corr_labels = self.corr_labels[:-1]
            return {
                "message": "Undid training sample",
                "success": True
            }
        else:
            return {
                "message": "Nothing to undo",
                "success": False
            }
        

    def add_sample_point(self, row, col, class_idx):
        self.corr_labels.append(class_idx)
        self.corr_features.append(self.current_features[row, col,:])
        return {
                "message": "Training sample for class %d added" % (class_idx),
                "success": True
        }
        
    def reset(self):
        self._init_model()
        self.corr_features = []
        self.corr_labels = []
        self.augment_model = sklearn.base.clone(SolarFineTuning.AUGMENT_MODEL)
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


    def run_model_on_tile(self, naip_tile, batch_size=16):
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
            t_batch = t_batch.type(torch.cuda.FloatTensor)
            
            with torch.no_grad():
                predictions, features = self.model.forward_features(t_batch)
                predictions = predictions.cpu().numpy()
                features = features.cpu().numpy()

            print(predictions.shape, features.shape)
            predictions = np.rollaxis(predictions, 1, 4)
            features = np.rollaxis(features, 1, 4)
            print(predictions.shape, features.shape)

            model_output.append(predictions)
            model_feature_output.append(features)

        model_output = np.concatenate(model_output, axis=0)
        model_feature_output = np.concatenate(model_feature_output, axis=0)
        
        for i, (y, x) in enumerate(batch_indices):
            output[y:y+self.input_size, x:x+self.input_size] += model_output[i] * kernel[..., np.newaxis]
            output_features[y:y+self.input_size, x:x+self.input_size] += model_feature_output[i] * kernel[..., np.newaxis]
            counts[y:y+self.input_size, x:x+self.input_size] += kernel

        output = output / counts[..., np.newaxis]
        output_features = output_features / counts[..., np.newaxis]

        return softmax(output), output_features

    def save_state_to(self, directory):
        
        
        return {
            "message": "Saved model state", 
            "success": True
        }

    def load_state_from(self, directory):
        
        
        return {
            "message": "Loaded model state", 
            "success": True
        }

    @property
    def last_tile(self):
        return 0