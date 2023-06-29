import logging
from sys import stderr
from functools import lru_cache

def getLogLevel(name: str) -> int:
    """Incredibly, i can't find a function that will do this conversion, other
    than setLevel, but setLevel doesn't work for calling the different log level logs."""
    if name.upper() in logging.getLevelNamesMapping():
        return logging.getLevelNamesMapping()[name.upper()]
    else:
        logging.log(logging.WARNING, "unknown log level level used: " + name + " assuming warning...")
        return logging.WARNING

def log(msg: str, loglevel: str = "INFO") -> None:
    level = getLogLevel(loglevel)
    logging.log(level, msg)


@lru_cache(None)
def log_once(msg: str, loglevel: str = "INFO") -> None:
    """A wrapper to log, to make sure that we only print things once"""
    log(msg, loglevel)


def setup_logger(outputfilename: str | None = None, loglevel: str = "INFO") -> None:
    """Sets up the logger with the necessary settings"""
    loggingformat = '[%(asctime)s] [Trainer] [%(levelname)s] %(message)s'
    if outputfilename is None:
        logging.basicConfig(stream=stderr, encoding='utf-8', level=getLogLevel(loglevel), format=loggingformat, datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(filename=outputfilename, encoding='utf-8', level=getLogLevel(loglevel), format=loggingformat, datefmt='%Y-%m-%d %H:%M:%S')
