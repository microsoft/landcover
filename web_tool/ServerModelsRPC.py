import sys, os, time
import rpyc

from log import LOGGER
from ServerModelsAbstract import BackendModel
from Utils import serialize, deserialize


class ModelRPC(BackendModel):

    def __init__(self, session_id, port):
        self.session_id = session_id

        self.max_retries = 30
        self.rety_timeout = 2
        self.connection = None

        i=0
        while self.connection is None or i<self.max_retries:
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
        
    def run(self, naip_data, extent, on_tile=False):
        return deserialize(self.connection.root.exposed_run(serialize(naip_data), extent, on_tile))
    def retrain(self):
        return self.connection.root.exposed_retrain()
    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        return self.connection.root.exposed_add_sample(tdst_row, bdst_row, tdst_col, bdst_col, class_idx)
    def undo(self):
        return self.connection.root.exposed_undo()
    def reset(self):
        return self.connection.root.exposed_reset()