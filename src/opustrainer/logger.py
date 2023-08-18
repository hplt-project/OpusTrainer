from io import TextIOWrapper
import logging
from sys import stderr, version_info
from typing import List, Dict, TextIO, Union, Optional
from functools import lru_cache

def _getLevelNamesMapping() -> Dict[str,int]:
    '''getLevelNamesMapping only available in python 3.11+'''
    if 'getLevelNamesMapping' in logging.__dict__:
        return logging.getLevelNamesMapping()
    else:
        return {'CRITICAL': 50,
                'FATAL': 50,
                'ERROR': 40,
                'WARN': 30,
                'WARNING': 30,
                'INFO': 20,
                'DEBUG': 10,
                'NOTSET': 0}

@lru_cache(None)
def get_log_level(name: str) -> int:
    """Incredibly, i can't find a function that will do this conversion, other
    than setLevel, but setLevel doesn't work for calling the different log level logs."""
    if name.upper() in _getLevelNamesMapping():
        return _getLevelNamesMapping()[name.upper()]
    else:
        logging.log(logging.WARNING, f"unknown log level level used: {name} assuming warning...")
        return logging.WARNING

def log(msg: str, loglevel: str = "INFO", **kwargs) -> None:
    level = get_log_level(loglevel)
    logging.log(level, msg, **kwargs)


@lru_cache(None)
def log_once(msg: str, loglevel: str = "INFO", **kwargs) -> None:
    """A wrapper to log, to make sure that we only print things once"""
    log(msg, loglevel, **kwargs)


def setup_logger(outputfilename: Optional[str] = None, loglevel: str = "INFO", disable_stderr: bool=False) -> None:
    """Sets up the logger with the necessary settings. Outputs to both file and stderr"""
    loggingformat = '[%(asctime)s] [Trainer] [%(levelname)s] %(message)s'
    handlers: List[Union[logging.StreamHandler[TextIO], logging.StreamHandler[TextIOWrapper]]] = []
    # disable_stderr is to be used only when testing the logger
    # When testing the logger directly, we don't want to write to stderr, because in order to read
    # our stderr output, we have to use redirect_stderr, which however makes all other tests spit
    # as it interferes with unittest' own redirect_stderr. How nice.
    # This happens even when assertLogs context capture is used.
    if not disable_stderr:
        handlers.append(logging.StreamHandler(stream=stderr))
    if outputfilename is not None:
        handlers.append(logging.FileHandler(filename=outputfilename))
    
    # Python 3.9 introduced an encoding argument
    if version_info[:2] >= (3,9):
        kwargs = {'encoding': 'utf-8'}
    else:
        kwargs = {}
    # This is the only logger we'd ever use. However during testing, due to the context, logger can't be recreated,
    # even if it has already been shutdown. This is why we use force=True to force recreation of logger so we can
    # properly run our tests. Not the best solution, not sure if it's not prone to race conditions, but it is
    # at the very least safe to use for the actual software running
    logging.basicConfig(handlers=handlers, level=get_log_level(loglevel), format=loggingformat,
                         datefmt='%Y-%m-%d %H:%M:%S', force=True, **kwargs) # type: ignore
