import sys
import os
import time
import datetime
import collections

import base64
import json
import uuid

import numpy as np

import pickle
import joblib
import pika

from Utils import get_random_string, AtomicCounter

from ServerModelsRPC import ModelRPC

from log import LOGGER


class Session():

    def __init__(self, session_id):
        LOGGER.info("Instantiating a new session object with id: %s" % (session_id))
        self.storage_type = None # this will be "table" or "file"
        self.storage_path = None # this will be a file path
        self.table_service = None # this will be an instance of TableService

        self.model = ModelRPC(session_id)
        self.current_transform = ()

        self.current_snapshot_string = get_random_string(8)
        self.current_snapshot_idx = 0
        self.current_request_counter = AtomicCounter()
        self.request_list = []
        

    def reset(self, soft=False, from_cached=None):
        if not soft:
            self.model.reset() # can't fail, so don't worry about it
        self.current_snapshot_string = get_random_string(8)
        self.current_snapshot_idx = 0
        self.current_request_counter = AtomicCounter()
        self.request_list = []

        if self.storage_type == "table":
            self.table_service.insert_entity("webtoolsessions",
            {
                "PartitionKey": str(np.random.randint(0,8)),
                "RowKey": str(uuid.uuid4()),
                "session_id": self.current_snapshot_string,
                "server_hostname": os.uname()[1],
                "server_sys_argv": ' '.join(sys.argv),
                "base_model": from_cached
            })

    def load(self, encoded_model_fn):
        model_fn = base64.b64decode(encoded_model_fn).decode('utf-8')

        print(model_fn)

        del self.model
        self.model = joblib.load(model_fn)

    def save(self, model_name):

        if self.storage_type is not None:
            assert self.storage_path is not None # we check for this when starting the program

            snapshot_id = "%s_%d" % (model_name, self.current_snapshot_idx)
            
            print("Saving state for %s" % (snapshot_id))
            base_dir = os.path.join(self.storage_path, self.current_snapshot_string)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=False)
            
            model_fn = os.path.join(base_dir, "%s_model.p" % (snapshot_id))
            joblib.dump(self.model, model_fn, protocol=pickle.HIGHEST_PROTOCOL)

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
    
    def add_entry(self, data):
        data = data.copy()
        data["time"] = datetime.datetime.now()
        data["current_snapshot_index"] = self.current_snapshot_idx
        current_request_counter = self.current_request_counter.increment()
        data["current_request_index"] = current_request_counter

        assert "experiment" in data

        if self.storage_type == "file":
            self.request_list.append(data)
        
        elif self.storage_type == "table":

            data["PartitionKey"] = self.current_snapshot_string
            data["RowKey"] = "%s_%d" % (data["experiment"], current_request_counter)

            for k in data.keys():
                if isinstance(data[k], dict) or isinstance(data[k], list):
                    data[k] = json.dumps(data[k])
            
            try:
                self.table_service.insert_entity("webtoolinteractions", data)
            except Exception as e:
                print(e)
        else:
            # The storage_type / --storage_path command line args were not set
            pass