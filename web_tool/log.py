import os
import time
from datetime import datetime

from pytz import timezone, utc

import logging
from logging.handlers import TimedRotatingFileHandler

DEFAULT_LOGGER_NAME = "logs"
LOGGER = logging.getLogger(DEFAULT_LOGGER_NAME)

def setup_logging(log_path, log_name, level=logging.DEBUG):
     
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # remove_old_logs(log_path) #TODO: reenable after `remove_old_logs()` is fixed 

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    logging.Formatter.converter = custom_time
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    logger.setLevel(level)
    
    fileHandler = TimedRotatingFileHandler(log_path + "/%s.txt" % (log_name), when='midnight', interval=1)
    fileHandler.suffix = "%Y%m%d"
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    
    printHandler = logging.StreamHandler()
    printHandler.setFormatter(formatter)
    logger.addHandler(printHandler)


def custom_time(*args):
    utc_dt = utc.localize(datetime.utcnow())
    my_tz = timezone("US/Pacific")
    converted = utc_dt.astimezone(my_tz)
    return converted.timetuple()

def remove_old_logs(log_path):
    try:
        now = time.time()
        for filename in os.listdir(log_path):
            if os.path.getmtime(os.path.join(log_path, filename)) < now - 7 * 86400: # TODO: this needs to check to make sure `filename` is a log file so it doesn't accidentally delete everything
                if os.path.isfile(os.path.join(log_path, filename)):
                    os.remove(os.path.join(log_path, filename))
        
    except Exception as e: #TODO: this needs to not capture all exceptions
        print(str(e))
            
