#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1137,E1136,E0110
import sys
import os
import time
import datetime
import collections
import argparse

import numpy as np

import rpyc
from rpyc.utils.server import OneShotServer, ThreadedServer

from ServerModelsKerasDense import KerasDenseFineTune
from Utils import serialize, deserialize

from log import setup_logging, LOGGER


class MyService(rpyc.Service):

    def __init__(self, model):
        self.model = model
        
    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass

    def exposed_run(self, naip_data, extent, on_tile=False):
        naip_data = deserialize(naip_data) # need to serialize/deserialize numpy arrays
        output = self.model.run(naip_data, extent, on_tile)
        return serialize(output) # need to serialize/deserialize numpy arrays

    def exposed_retrain(self):
        return self.model.retrain()

    def exposed_add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        return self.model.add_sample(tdst_row, bdst_row, tdst_col, bdst_col, class_idx)

    def exposed_undo(self):
        return self.model.undo()

    def exposed_reset(self):
        return self.model.reset()

def main():
    global MODEL
    parser = argparse.ArgumentParser(description="AI for Earth Land Cover Worker")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)

    parser.add_argument("--port", action="store", type=int, help="Port we are listenning on", default=0)
    parser.add_argument("--model", action="store", dest="model",
        choices=[
            "keras_dense"
        ],
        help="Model to use", required=True
    )
    parser.add_argument("--model_fn", action="store", dest="model_fn", type=str, help="Model fn to use", default=None)
    parser.add_argument("--fine_tune_layer", action="store", dest="fine_tune_layer", type=int, help="Layer of model to fine tune", default=-2)
    parser.add_argument("--fine_tune_seed_data_fn", action="store", dest="fine_tune_seed_data_fn", type=str, help="Path to npz containing seed data to use", default=None)
    
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", required=True)

    args = parser.parse_args(sys.argv[1:])

    # Setup logging
    log_path = os.getcwd() + "/logs"
    setup_logging(log_path, "worker")


    # Setup model
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpuid is None else str(args.gpuid)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    if args.model == "keras_dense":
        model = KerasDenseFineTune(args.model_fn, args.gpuid, args.fine_tune_layer, args.fine_tune_seed_data_fn)
    else:
        raise NotImplementedError("The given model type is not implemented yet.")

    t = OneShotServer(MyService(model), port=args.port)
    t.start()
   
if __name__ == "__main__":
    main()