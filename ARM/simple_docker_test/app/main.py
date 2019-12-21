import logging
import subprocess
import shlex

log = "/app/logs/docker.log"

logging.basicConfig(filename=log,level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logging.info('testing arm docker image deployment')

subprocess.call(shlex.split('bash deallocate_vm.sh'))
