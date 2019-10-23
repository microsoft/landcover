import sys, os, time

import uuid
import pika
import pickle

from ServerModelsAbstract import BackendModel


class ModelRPC(BackendModel):

    def __init__(self, session_id):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')
        )

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self._on_response,
            auto_ack=True
        )

        self.correlation_ids = set()
        self.responses = dict()
     

    def _on_response(self, ch, method, props, body):
        if props.correlation_id in self.correlation_ids:
            self.responses[props.correlation_id] = pickle.loads(body, encoding="bytes")
            self.correlation_ids.remove(props.correlation_id)


    def _call(self, method_name, args):

        correlation_id = str(uuid.uuid4()) 
        self.correlation_ids.add(correlation_id)
        self.responses[correlation_id] = None

        self.channel.basic_publish(
            exchange='',
            routing_key='rpc_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=correlation_id,
            ),
            body=pickle.dumps((method_name, args))
        )

        while self.responses[correlation_id] is None:
            self.connection.process_data_events()
        
        local = self.responses[correlation_id] 
        del self.responses[correlation_id]

        return local


    def run(self, naip_data, extent, on_tile=False):
        return self._call("run", (naip_data, extent, on_tile))

    def retrain(self, **kwargs):
        return self._call("retrain", (kwargs))
        
    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        self._call("add_sample", (tdst_row, bdst_row, tdst_col, bdst_col, class_idx))

    def undo(self):
        return self._call("undo", ())

    def reset(self):
        return self._call("reset", ())