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

import pickle

from ServerModelsKerasDense import KerasDenseFineTune

from log import setup_logging, LOGGER
import pika


def main():
    parser = argparse.ArgumentParser(description="AI for Earth Land Cover Worker")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)


    # TODO: add support for consuming work for a single sessionID
    parser.add_argument("--session-id", action="store", type=str, help="Name of session we are listenning to", default="*")

    parser.add_argument("--host", action="store", dest="host", type=str, help="RabbitMQ server", default="localhost")
    parser.add_argument("--user",  action="store", type=str, help="RabbitMQ server username", default="guest")
    parser.add_argument("--password",  action="store", type=str, help="RabbitMQ server password", default="guest")

    parser.add_argument("--model", action="store", dest="model",
        choices=[
            "keras_dense"
        ],
        help="Model to use", required=True
    )
    parser.add_argument("--fine_tune_seed_data_fn", action="store", dest="fine_tune_seed_data_fn", type=str, help="Path to npz containing seed data to use", default=None)
    parser.add_argument("--fine_tune_layer", action="store", dest="fine_tune_layer", type=int, help="Layer of model to fine tune", default=-2)
    parser.add_argument("--model_fn", action="store", dest="model_fn", type=str, help="Model fn to use", default=None)
    
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", default=None)

    args = parser.parse_args(sys.argv[1:])

    # Setup logging
    log_path = os.getcwd() + "/logs"
    setup_logging(log_path, "worker")


    # Setup model
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpuid is None else str(args.gpuid)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    model = None
    if args.model == "keras_dense":
        model = KerasDenseFineTune(args.model_fn, args.gpuid, args.fine_tune_layer, args.fine_tune_seed_data_fn)
    else:
        raise NotImplementedError("The given model type is not implemented yet.")


    credentials = pika.PlainCredentials(args.user, args.password)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=args.host, credentials=credentials)
    )

    channel = connection.channel()

    channel.exchange_declare(exchange='rpc_exchange', exchange_type='direct')

    result = channel.queue_declare(queue=args.session_id, exclusive=False)
    queue_name = result.method.queue
    channel.queue_bind(exchange='rpc_exchange', queue=queue_name, routing_key=args.session_id)

    def on_request(ch, method, props, body):
        method_name, args = pickle.loads(body)

        LOGGER.info("Received request: %s" % (method_name))

        response = None
        if method_name == "run":
            response = model.run(*args)
        elif method_name == "retrain":
            response = model.retrain()
        elif method_name == "add_sample":
            response = model.add_sample(*args)
        elif method_name == "reset":
            response = model.reset()
        elif method_name == "undo":
            response = model.undo()
        else:
            raise ValueError("Invalid request received")

        if response is None: # For some reason I don't understand, if we don't return actual data then "process_data_events()" in ServerModelsRPC.py will hang
            response = 1

        LOGGER.info("Before publish")

        ch.basic_publish(
            exchange='rpc_exchange',
            routing_key=props.reply_to,
            properties=pika.BasicProperties(
                correlation_id=props.correlation_id
            ),
            body=pickle.dumps(response)
        )

        LOGGER.info("After publish")

        ch.basic_ack(delivery_tag=method.delivery_tag)

        LOGGER.info("After ack")


    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=queue_name, on_message_callback=on_request)

    LOGGER.info("Awaiting requests")
    channel.start_consuming()

if __name__ == "__main__":
    main()