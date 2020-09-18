import sys
import os
import time
import datetime
import collections
import subprocess
import shutil

import base64
import json
import uuid
import pickle

import numpy as np

import joblib

from .Utils import get_random_string, AtomicCounter
from .Checkpoints import Checkpoints
from .DataLoader import InMemoryRaster

import logging
LOGGER = logging.getLogger("server")

SESSION_BASE_PATH = './tmp/session'
SESSION_FOLDER = SESSION_BASE_PATH + "/" + datetime.datetime.now().strftime('%Y-%m-%d')


def manage_session_folders():
    if not os.path.exists(SESSION_BASE_PATH):
        os.makedirs(SESSION_BASE_PATH)
    if not os.path.exists(SESSION_FOLDER):
        shutil.rmtree(SESSION_BASE_PATH)
        os.makedirs(SESSION_FOLDER)


class Session():

    def __init__(self, session_id, model):
        LOGGER.info("Instantiating a new session object with id: %s" % (session_id))

        self.model = model
        self.data_loader = None
        self.latest_input_raster = None # InMemoryRaster object from the most recent prediction
        self.tile_map = None # A map recording the most recent prediction per pixel

        self.current_snapshot_string = get_random_string(8)
        self.current_snapshot_idx = 0
        self.current_request_counter = AtomicCounter()
        self.request_list = []

        self.session_id = session_id
        self.creation_time = time.time()
        self.last_interaction_time = self.creation_time

    def reset(self):
        self.current_snapshot_string = get_random_string(8)
        self.current_snapshot_idx = 0
        self.current_request_counter = AtomicCounter()
        self.request_list = []
        return self.model.reset()

    def load(self, encoded_model_fn):
        model_fn = base64.b64decode(encoded_model_fn).decode('utf-8')

        del self.model
        self.model = joblib.load(model_fn)

    def save(self, model_name):

        if self.storage_type is not None:
            assert self.storage_path is not None # we check for this when starting the program

            snapshot_id = "%s_%d" % (model_name, self.current_snapshot_idx)
            
            LOGGER.info("Saving state for %s" % (snapshot_id))
            base_dir = os.path.join(self.storage_path, self.current_snapshot_string)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=False)
            
            model_fn = os.path.join(base_dir, "%s_model.p" % (snapshot_id))
            #joblib.dump(self.model, model_fn, protocol=pickle.HIGHEST_PROTOCOL)

            if self.storage_type == "file":
                request_list_fn = os.path.join(base_dir, "%s_request_list.p" % (snapshot_id))
                joblib.dump(self.request_list, request_list_fn, protocol=pickle.HIGHEST_PROTOCOL)
            elif self.storage_type == "table":
                # We don't serialize the request list when saving to table storage
                pass

            self.current_snapshot_idx += 1
            return base64.b64encode(model_fn.encode('utf-8')).decode('utf-8') # this is super dumb
        else:
            return None


    def create_checkpoint(self, dataset_name, model_name, checkpoint_name, classes):
        
        if "-" in checkpoint_name:
            return {
                "message": "Checkpoint name cannot contain '-'",
                "success": False
            }
        elif checkpoint_name == "new":
            return {
                "message": "Checkpoint name cannot be 'new'",
                "success": False
            }


        try:
            directory = Checkpoints.create_new_checkpoint_directory(dataset_name, model_name, checkpoint_name)
        except ValueError as e:
            return {
                "message": e.args[0],
                "success": False
            }

        with open(os.path.join(directory, "classes.json"), "w") as f:
            f.write(json.dumps(classes))

        return self.model.save_state_to(directory)

    def add_entry(self, data):
        # data = data.copy()
        # data["time"] = datetime.datetime.now()
        # data["current_snapshot_index"] = self.current_snapshot_idx
        # current_request_counter = self.current_request_counter.increment()
        # data["current_request_index"] = current_request_counter

        # assert "experiment" in data

        # if self.storage_type == "file":
        #     self.request_list.append(data)
        
        # elif self.storage_type == "table":

        #     data["PartitionKey"] = self.current_snapshot_string
        #     data["RowKey"] = "%s_%d" % (data["experiment"], current_request_counter)

        #     for k in data.keys():
        #         if isinstance(data[k], dict) or isinstance(data[k], list):
        #             data[k] = json.dumps(data[k])
            
        #     try:
        #         self.table_service.insert_entity("webtoolinteractions", data)
        #     except Exception as e:
        #         LOGGER.error(e)
        # else:
        #     # The storage_type / --storage_path command line args were not set
        #     pass
        pass

    def pred_patch(self, input_raster):
        output = self.model.run(input_raster.data, False)
        assert input_raster.shape[0] == output.shape[0] and input_raster.shape[1] == output.shape[1], "ModelSession must return an np.ndarray with the same height and width as the input"

        return InMemoryRaster(output, input_raster.crs, input_raster.transform, input_raster.bounds)

    def pred_tile(self, input_raster):
        output = self.model.run(input_raster.data, True)
        assert input_raster.shape[0] == output.shape[0] and input_raster.shape[1] == output.shape[1], "ModelSession must return an np.ndarray with the same height and width as the input"

        return InMemoryRaster(output, input_raster.crs, input_raster.transform, input_raster.bounds)

    def download_all(self):
        pass

