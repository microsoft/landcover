import logging
import subprocess

log = "/app/logs/docker.log"

logging.basicConfig(filename=log,level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logging.info('testing arm docker image deployment')

subprocess.call(["sudo", "apt", "update"])
