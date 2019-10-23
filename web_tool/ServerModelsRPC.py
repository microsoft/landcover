import sys, os, time

import uuid
import pika
import pickle

from log import LOGGER

import threading

from ServerModelsAbstract import BackendModel


class ModelRPC(BackendModel):

    def __init__(self, session_id):

        self.thread_ids = set()
        self.channels = dict()
        self.connections = dict()
        self.callback_queues = dict()

        self.correlation_ids = set()
        self.responses = dict()     

    def _on_response(self, ch, method, props, body):
        if props.correlation_id in self.correlation_ids:
            LOGGER.info("Finished %s" % (props.correlation_id))
            self.responses[props.correlation_id] = pickle.loads(body, encoding="bytes")
            self.correlation_ids.remove(props.correlation_id)

    def _register_new_thread(self, thread_id):

        self.connections[thread_id] = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )

        self.channels[thread_id] = self.connections[thread_id].channel()

        result = self.channels[thread_id].queue_declare(queue='', exclusive=True)
        self.callback_queues[thread_id] = result.method.queue

        self.channels[thread_id].basic_consume(
            queue=self.callback_queues[thread_id],
            on_message_callback=self._on_response,
            auto_ack=True
        )

        self.thread_ids.add(thread_id)

    def _check_thread_connection(self, thread_id):
        try:
            #self.connections[thread_id].sleep(0)
            self.connections[thread_id].process_data_events(0)
            return True
        except pika.exceptions.StreamLostError as e:
            LOGGER.info("Thread (%d) died because %s, remaking it" % (thread_id, str(e)))
            return False

    def _call(self, method_name, args):

        thread_id = threading.get_ident()
        if thread_id not in self.thread_ids or not self._check_thread_connection(thread_id):
            LOGGER.info("Registering new connection for thread (%d)" % (thread_id))
            self._register_new_thread(thread_id)

        correlation_id = str(uuid.uuid4()) 
        self.correlation_ids.add(correlation_id)
        self.responses[correlation_id] = None

        LOGGER.info("Publishing %s with %s" % (method_name, correlation_id))

        self.channels[thread_id].basic_publish(
            exchange='',
            routing_key='rpc_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queues[thread_id],
                correlation_id=correlation_id,
            ),
            body=pickle.dumps((method_name, args))
        )

        while self.responses[correlation_id] is None:
            self.connections[thread_id].process_data_events()
        
        local = self.responses[correlation_id] 
        del self.responses[correlation_id]

        LOGGER.info("Exiting _call %s (%d active)" % (correlation_id, len(self.responses)))
        return local


    def run(self, naip_data, extent, on_tile=False):
        return self._call("run", (naip_data, extent, on_tile))

    def retrain(self):
        return self._call("retrain", ())
        
    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        self._call("add_sample", (tdst_row, bdst_row, tdst_col, bdst_col, class_idx))

    def undo(self):
        return self._call("undo", ())

    def reset(self):
        return self._call("reset", ())