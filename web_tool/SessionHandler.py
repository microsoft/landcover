import time
import threading
import subprocess
import socket
from queue import Queue

from Session import Session

from ServerModelsRPC import ModelRPC
from ServerModelsKerasDense import KerasDenseFineTune

from log import LOGGER

from Models import load_models
MODELS = load_models()


def session_monitor(session_handler, session_timeout_seconds=900):
    ''' This is a `Thread()` that is starting when the program is run. It is responsible for finding which of the `Session()` objects
    in `SESSION_MAP` haven't been used recently and killing them.

    TODO: SESSION_HANDLER needs to be thread safe because we call it from here and the main sever threads
    '''
    LOGGER.info("Starting session monitor thread")
    while True:
        session_ids_to_kill = []
        for session_id, session in session_handler._SESSION_MAP.items():
            #LOGGER.info("SESSION MONITOR - Checking session (%s) for activity" % (session_id))
            time_inactive = time.time() - session.last_interaction_time
            if time_inactive > session_timeout_seconds:
                session_ids_to_kill.append(session_id)
        
        for session_id in session_ids_to_kill:
            LOGGER.info("SESSION MONITOR - Session (%s) has been inactive for over %d seconds, destroying" % (session_id, session_timeout_seconds))
            session_handler.kill_session(session_id)
        
        time.sleep(5)


def get_free_tcp_port():
    '''From: https://gist.github.com/gabrielfalcao/20e567e188f588b65ba2
    
    We use the OS to select a free port for us.
    '''
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(('', 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


class SessionHandler():

    def __init__(self, local, args):
        self._WORKERS = [ # TODO: I hardcode that there are 4 GPUs available on the local machine
            {"type": "local", "gpu_id": 0},
            {"type": "local", "gpu_id": 1},
            {"type": "local", "gpu_id": 2},
            {"type": "local", "gpu_id": 3}
        ]

        self._WORKER_POOL = Queue()
        for worker in self._WORKERS:
            self._WORKER_POOL.put(worker)

        self._expired_sessions = set()
        self._SESSION_MAP = dict()
        self._SESSION_INFO = dict()
        
        self.local = local
        self.args = args


    def is_active(self, session_id):
        return session_id in self._SESSION_MAP


    def is_expired(self, session_id):
        '''This is checked in `manage_sessions` in server.py before each request.
        
        If the user's current session_id is marked as expired, then it has been deleted
        and `manage_sessions` will delete the session content on the actual client side
        (I think by destroying the users's cookie).

        A session_id is ONLY marked as expired by `kill_session`.
        '''
        return session_id in self._expired_sessions


    def _set_expired(self, session_id):
        self._expired_sessions.add(session_id)


    def cleanup_expired_session(self, session_id):
        '''After `manage_sessions` cleans up the session on the client side, then
        the session_id can be removed from the expired session set.
        ''' 
        self._expired_sessions.remove(session_id)


    def _spawn_local_worker(self, port, model_fn, gpu_id, fine_tune_layer):
        command = [
            "/usr/bin/env", "python", "web_tool/worker.py",
            "--model", "keras_dense",
            "--model_fn", model_fn,
            "--fine_tune_layer", str(fine_tune_layer),
            "--gpu", str(gpu_id),
            "--port", str(port)
        ]
        process = subprocess.Popen(command, shell=False)
        return process


    def create_session(self, session_id, model_key):
        if session_id in self._SESSION_MAP:
            raise ValueError("session_id %d has already been created" % (session_id))

        if model_key not in MODELS:
            raise ValueError("%s is not a valid model, check the keys in models.json" % (model_key))
        
        model_fn = MODELS[model_key]["fn"]
        fine_tune_layer = MODELS[model_key]["fine_tune_layer"]

        worker = self._WORKER_POOL.get() # this will block until we have a free one

        if worker["type"] == "local":
            random_port = get_free_tcp_port()
            gpu_id = worker["gpu_id"]
            process = self._spawn_local_worker(random_port, model_fn, gpu_id, fine_tune_layer)
            model = ModelRPC(session_id, random_port)
            session = Session(session_id, model)
            self._SESSION_MAP[session_id] = session
            self._SESSION_INFO[session_id] = {
                "worker": worker,
                "process": process
            }
            LOGGER.info("Created a local worker for (%s) on GPU %d" % (session_id, gpu_id))

        elif worker["type"] == "remote":
            raise NotImplementedError("Remote workers aren't implemented yet")
        else:
            raise ValueError("Worker type %s isn't recognized" % (worker["type"]))


    def kill_session(self, session_id):
        ''' Two code paths should be able to kill a session:
        - The user kills the session on purpose through /killSession
        - The `session_monitor` thread sees that the session hasn't had any activity for some time 
        '''
        if self.is_active(session_id):
            # kill the remote process
            self._SESSION_INFO[session_id]["process"].kill()
            # add the worker back into the worker pool
            self._WORKER_POOL.put(self._SESSION_INFO[session_id]["worker"])
            del self._SESSION_INFO[session_id]

            # TODO: is there anything that needs to be cleaned up at the session level (e.g. saving data)?
            del self._SESSION_MAP[session_id]
            
            self._set_expired(session_id) # we set this to expired so that it can be cleaned up on the client side
        else:
            raise ValueError("Tried to kill a non-existing Session")


    def get_session(self, session_id):
        if self.is_active(session_id):
            return self._SESSION_MAP[session_id]
        else:
            raise ValueError("Tried to get a non-existing Session")


    def touch_session(self, session_id):
        if self.is_active(session_id):
            self._SESSION_MAP[session_id].last_interaction_time = time.time()
        else:
            raise ValueError("Tried to update time on a non-existing Session")


    def start_monitor(self, session_timeout_seconds=900):
        session_monitor_thread = threading.Thread(target=session_monitor, args=(self, session_timeout_seconds))
        session_monitor_thread.setDaemon(True)
        session_monitor_thread.start()