"""
Logging.
"""

from enum import Enum
from logging import Formatter, Logger, StreamHandler, getLevelName
from sys import stdout

STREAM_HANDLER = StreamHandler(stream=stdout)
STREAM_FORMATTER = Formatter("%(name)s - %(levelname)s - %(message)s")
STREAM_HANDLER.setFormatter(STREAM_FORMATTER)


class LogLevel(str, Enum):
    """
    Log level enum.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


def get_logger(name: str) -> Logger:
    """
    Get a logger. Attaches to a global stream handler to stdout by default.

    Args:
        name: The name of the logger.

    Returns:
        A logger.

    Examples:
        >>> logger = get_logger("my_logger")
        >>> logger.info("Hello world!")
        my_logger - INFO - Hello world!
    """
    logger = Logger(name=name)
    logger.addHandler(STREAM_HANDLER)
    return logger


def set_stream_handler_level(level: LogLevel):
    """
    Set the level of the global stream handler.

    Args:
        level: The level to set.

    Examples:
        >>> set_stream_handler_level(LogLevel.DEBUG)
    """
    STREAM_HANDLER.setLevel(getLevelName(level))
