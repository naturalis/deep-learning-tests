import os, logging


class LogClass:

    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        logfile = os.path.join(os.environ['PROJECT_ROOT'], 'log', 'general.log')

        # if not os.path.exists(logfile):
        #     logfile = 'general.log'

        fh = logging.FileHandler(logfile)

        if 'DEBUGGING' in os.environ and os.environ['DEBUGGING'] == "1":
            fh.setLevel(logging.DEBUG)
        else:
            fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def __del__(self):
        pass

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)
