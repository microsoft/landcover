import os

import pickle
import joblib
import numpy as np

import sklearn.base
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
import tensorflow.keras as keras

import logging
LOGGER = logging.getLogger("server")

from . import ROOT_DIR
from .ModelSessionAbstract import ModelSession

class ModelSessionRandomForest(ModelSession):

    AUGMENT_MODEL = RandomForestClassifier()

    def __init__(self, **kwargs):

        self.augment_x_train = []
        self.augment_y_train = []
        self.augment_model = sklearn.base.clone(ModelSessionRandomForest.AUGMENT_MODEL)
        self.augment_model_trained = False
        
        self._last_tile = None
     
    @property
    def last_tile(self):
        return self._last_tile

    def run(self, tile, inference_mode=False):
        tile = tile / 256.0

        if self.augment_model_trained:
            original_shape = tile.shape
            output = tile.reshape(-1, tile.shape[2])
            output = self.augment_model.predict_proba(output)
            output = output.reshape(original_shape[0], original_shape[1], -1)
        else:
            output = tile.copy()

        if not inference_mode:
            self._last_tile = tile

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

            self.augment_model_trained = True
            
            return {
                "message": "Fine-tuning accuracy on data: %0.2f" % (score),
                "success": True
            }
        except Exception as e:
            return {
                "message": "Error in 'retrain()': %s" % (e),
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

    def reset(self):
        self._last_tile = None
        self.augment_x_train = []
        self.augment_y_train = []
        self.augment_model = sklearn.base.clone(ModelSessionRandomForest.AUGMENT_MODEL)
        self.augment_model_trained = False

        return {
            "message": "Model reset successfully",
            "success": True
        }

    def save_state_to(self, directory):
        
        np.save(os.path.join(directory, "augment_x_train.npy"), np.array(self.augment_x_train))
        np.save(os.path.join(directory, "augment_y_train.npy"), np.array(self.augment_y_train))

        joblib.dump(self.augment_model, os.path.join(directory, "augment_model.p"))

        if self.augment_model_trained:
            with open(os.path.join(directory, "trained.txt"), "w") as f:
                f.write("")

        return {
            "message": "Saved model state", 
            "success": True
        }

    def load_state_from(self, directory):

        self.augment_x_train = []
        self.augment_y_train = []

        for sample in np.load(os.path.join(directory, "augment_x_train.npy")):
            self.augment_x_train.append(sample)
        for sample in np.load(os.path.join(directory, "augment_y_train.npy")):
            self.augment_y_train.append(sample)

        self.augment_model = joblib.load(os.path.join(directory, "augment_model.p"))
        self.augment_model_trained = os.path.exists(os.path.join(directory, "trained.txt"))

        return {
            "message": "Loaded model state", 
            "success": True
        }
