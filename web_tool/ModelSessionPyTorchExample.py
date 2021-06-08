import sys
sys.path.append("..")

import os
import time
import copy
import json
import types

import logging
LOGGER = logging.getLogger("server")

import numpy as np

import sklearn.base
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

from .ModelSessionAbstract import ModelSession
from training.models.unet_solar import UnetModel
from training.models.fcn import FCN

import torch
import torch.nn as nn
import torch.nn.functional as F

    
class TorchFineTuning(ModelSession):

    AUGMENT_MODEL = MLPClassifier(
        hidden_layer_sizes=(),
        alpha=0.0001,
        solver='lbfgs',
        tol=0.0001,
        verbose=False,
        validation_fraction=0.0,
        n_iter_no_change=50
    )


    def __init__(self, gpu_id, **kwargs):
        self.model_fn = kwargs["fn"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.output_channels = kwargs["num_classes"]
        self.output_features = 64
        self.input_size = kwargs["input_size"]
        self.input_channels = kwargs["input_channels"]

        self.down_weight_padding = 10

        self.stride_x = self.input_size - self.down_weight_padding*2
        self.stride_y = self.input_size - self.down_weight_padding*2

        self.model = FCN(self.input_channels, num_output_classes=self.output_channels, num_filters=64)
        self._init_model()

        for param in self.model.parameters():
            param.requires_grad = False

        self.augment_model = sklearn.base.clone(TorchFineTuning.AUGMENT_MODEL)

        self._last_tile = None

        with np.load(kwargs["seed_data_fn"]) as f:
            embeddings = f["embeddings"].copy()
            labels = f["labels"].copy()

            idxs = np.random.choice(embeddings.shape[0], size=500)

            self.augment_x_base = embeddings[idxs]
            self.augment_y_base = labels[idxs]

        self.augment_x_train = []
        self.augment_y_train = []
        for row in self.augment_x_base:
            self.augment_x_train.append(row)
        for row in self.augment_y_base:
            self.augment_y_train.append(row)

    @property
    def last_tile(self):
        return self._last_tile

    def _init_model(self):
        checkpoint = torch.load(self.model_fn, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model = self.model.to(self.device)

    def run(self, tile, inference_mode=False):
        if tile.shape[2] == 3: # If we get a 3 channel image, then pretend it is 4 channel by duplicating the first band
            tile = np.concatenate([
                tile,
                tile[:,:,0][:,:,np.newaxis]
            ], axis=2)

        tile = tile / 255.0
        tile = tile.astype(np.float32)

        output, output_features = self.run_model_on_tile(tile)
        self._last_tile = output_features

        return output

    def retrain(self, **kwargs):
        x_train = np.array(self.augment_x_train)
        y_train = np.array(self.augment_y_train)

        if x_train.shape[0] == 0:
            return {
                "message": "Need to add training samples in order to train",
                "success": False
            }


        try:
            self.augment_model.fit(x_train, y_train)
            score = self.augment_model.score(x_train, y_train)
            LOGGER.debug("Fine-tuning accuracy: %0.4f" % (score))

            new_weights = torch.from_numpy(self.augment_model.coefs_[0].T.copy().astype(np.float32)[:,:,np.newaxis,np.newaxis])
            new_biases = torch.from_numpy(self.augment_model.intercepts_[0].astype(np.float32))
            new_weights = new_weights.to(self.device)
            new_biases = new_biases.to(self.device)

            self.model.last.weight.data = new_weights
            self.model.last.bias.data = new_biases
            
            return {
                "message": "Fine-tuning accuracy on data: %0.2f" % (score),
                "success": True
            }
        except Exception as e:
            return {
                "message": "Error in 'retrain()': %s" % (e),
                "success": False
            }
    
    def undo(self):
        if len(self.augment_y_train) > 0:
            self.augment_x_train.pop()
            self.augment_y_train.pop()
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
        if self._last_tile is not None:
            self.augment_x_train.append(self._last_tile[row, col, :].copy())
            self.augment_y_train.append(class_idx)
            return {
                "message": "Training sample for class %d added" % (class_idx),
                "success": True
            }
        else:
            return {
                "message": "Must run model before adding a training sample",
                "success": False
            }
        
    def reset(self):
        self._init_model()
        self.augment_model = sklearn.base.clone(TorchFineTuning.AUGMENT_MODEL)

        self.augment_x_train = []
        self.augment_y_train = []
        for row in self.augment_x_base:
            self.augment_x_train.append(row)
        for row in self.augment_y_base:
            self.augment_y_train.append(row)

        return {
            "message": "Model reset successfully",
            "success": True
        }


    def run_model_on_tile(self, tile, batch_size=32):
        height = tile.shape[0]
        width = tile.shape[1]
        
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
                naip_im = tile[y_index:y_index+self.input_size, x_index:x_index+self.input_size, :]

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
                predictions = F.softmax(predictions)

                predictions = predictions.cpu().numpy()
                features = features.cpu().numpy()

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
        output_features = output_features / counts[..., np.newaxis]

        return output, output_features

    def save_state_to(self, directory):
        raise NotImplementedError()
        # np.save(os.path.join(directory, "augment_x_train.npy"), np.array(self.augment_x_train))
        # np.save(os.path.join(directory, "augment_y_train.npy"), np.array(self.augment_y_train))

        # joblib.dump(self.augment_model, os.path.join(directory, "augment_model.p"))

        # if self.augment_model_trained:
        #     with open(os.path.join(directory, "trained.txt"), "w") as f:
        #         f.write("")

        return {
            "message": "Saved model state", 
            "success": True
        }

    def load_state_from(self, directory):
        raise NotImplementedError()
        # self.augment_x_train = []
        # self.augment_y_train = []

        # for sample in np.load(os.path.join(directory, "augment_x_train.npy")):
        #     self.augment_x_train.append(sample)
        # for sample in np.load(os.path.join(directory, "augment_y_train.npy")):
        #     self.augment_y_train.append(sample)

        # self.augment_model = joblib.load(os.path.join(directory, "augment_model.p"))
        # self.augment_model_trained = os.path.exists(os.path.join(directory, "trained.txt"))

        return {
            "message": "Loaded model state", 
            "success": True
        }

    # def retrain(self, train_steps=100, learning_rate=1e-3):
      
    #     print_every_k_steps = 10

    #     print("Fine tuning with %d new labels." % self.num_corrected_pixels)
    #     batch_x = torch.from_numpy(np.array(self.augment_x_train)).float().to(self.device)
    #     batch_y = torch.from_numpy(np.array(self.augment_y_train)).to(self.device)
        
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
    #     message = "Fine-tuned model with %d samples." % len(self.augment_x_train)
    #     print(message)
    #     return success, message