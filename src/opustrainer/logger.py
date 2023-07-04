from io import TextIOWrapper
import logging
from sys import stderr
from typing import List, TextIO
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


def setup_logger(outputfilename: str | None = None, loglevel: str = "INFO", disable_stderr: bool=False) -> None:
    """Sets up the logger with the necessary settings. Outputs to both file and stderr"""
    loggingformat = '[%(asctime)s] [Trainer] [%(levelname)s] %(message)s'
    handlers: List[logging.StreamHandler[TextIO] | logging.StreamHandler[TextIOWrapper]] = []
    # disable_stderr is to be used only when testing the logger
    # When testing the logger directly, we don't want to write to stderr, because in order to read
    # our stderr output, we have to use redirect_stderr, which however makes all other tests spit
    # as it interferes with unittest' own redirect_stderr. How nice.
    if not disable_stderr:
        handlers.append(logging.StreamHandler(stream=stderr))
    if outputfilename is not None:
        handlers.append(logging.FileHandler(filename=outputfilename))
    logging.basicConfig(handlers=handlers, encoding='utf-8', level=getLogLevel(loglevel), format=loggingformat, datefmt='%Y-%m-%d %H:%M:%S')

