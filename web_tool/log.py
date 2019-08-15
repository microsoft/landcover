import os
import logging
import time

from datetime import datetime
from pytz import timezone, utc
from logging.handlers import TimedRotatingFileHandler

class Log():
    def __init__(self, log_path, name='log', level=logging.DEBUG):
        self.log_path = log_path
        
        if not os.path.exists(self.log_path):
            print(self.log_path)
            os.mkdir(self.log_path)

        self.remove_old_logs()

        handler = TimedRotatingFileHandler(self.log_path + "/log.log", when='midnight', interval=1)
        handler.suffix = "%Y%m%d"

        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

        self.logger = logging.getLogger(name)
        self.logger.addHandler(handler)
        self.logger.setLevel(level)
        logging.Formatter.converter = self.customTime


    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def customTime(self, *args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone("US/Pacific")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()

    def remove_old_logs(self):
        try:
            now = time.time()

            for filename in os.listdir(self.log_path):
                if os.path.getmtime(os.path.join(self.log_path, filename)) < now - 7 * 86400:
                    if os.path.isfile(os.path.join(self.log_path, filename)):
                        print(filename)
                        os.remove(os.path.join(self.log_path, filename))
            
        except Exception as e:
            print(str(e))
            
