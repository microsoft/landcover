import sys, os, time
import rpyc

import logging
LOGGER = logging.getLogger("server")

from .ModelSessionAbstract import ModelSession
from .Utils import serialize, deserialize

def clean_output_dict(data):
    return {
        "message": data["message"],
        "success": data["success"]
    }

class ModelSessionRPC(ModelSession):

    def __init__(self, gpu_id, **kwargs):

        session_id = kwargs["session_id"]
        port = kwargs["port"]

        self.session_id = session_id

        self.max_retries = 30
        self.rety_timeout = 2
        self.connection = None

        i=0
        while self.connection is None and i<self.max_retries:
            try:
                self.connection = rpyc.connect("localhost", port, config={
                    'allow_public_attrs': False
                })
                LOGGER.info("Made a connection")

                if "load_dir" in kwargs:
                    self.load_state_from(kwargs["load_dir"])

                break
            except ConnectionRefusedError:
                i+=1
                time.sleep(self.rety_timeout)
                LOGGER.warning("Haven't connected, attempt %d" % (i))
    @property
    def last_tile(self):
        return deserialize(self.connection.root.exposed_last_tile)
    def run(self, tile, inference_mode):
        return deserialize(self.connection.root.exposed_run(serialize(tile), inference_mode))
    def retrain(self):
        return clean_output_dict(self.connection.root.exposed_retrain())
    def add_sample_point(self, row, col, class_idx):
        return clean_output_dict(self.connection.root.exposed_add_sample_point(row, col, class_idx))
    def undo(self):
        return clean_output_dict(self.connection.root.exposed_undo())
    def reset(self):
        return clean_output_dict(self.connection.root.exposed_reset())
    def save_state_to(self, directory):
        return clean_output_dict(self.connection.root.exposed_save_state_to(directory))
    def load_state_from(self, directory):
        return clean_output_dict(self.connection.root.exposed_load_state_from(directory))