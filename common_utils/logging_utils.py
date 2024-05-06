import logging
from sys import stdout


class LoggingUtilities():

    def __init__(self):

        pass

    @staticmethod
    def console_out_logger(name):

        # set up/configure logging with stdout so it can be picked up by K8s
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(message)s')  # noqa: E501
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

        return logger
