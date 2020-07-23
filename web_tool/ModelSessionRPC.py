import sys, os, time
import rpyc

import logging
LOGGER = logging.getLogger("server")

from .ModelSessionAbstract import ModelSession
from .Utils import serialize, deserialize


class ModelSessionRPC(ModelSession):

    def __init__(self, session_id, port):
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
        return self.connection.root.exposed_retrain()
    def add_sample_point(self, row, col, class_idx):
        return self.connection.root.exposed_add_sample_point(row, col, class_idx)
    def undo(self):
        return self.connection.root.exposed_undo()
    def reset(self):
        return self.connection.root.exposed_reset()
    def save_state_to(self, directory):
        return self.connection.root.exposed_save_state_to(directory)
    def load_state_from(self, directory):
        return self.connection.root.exposed_load_state_from(directory)