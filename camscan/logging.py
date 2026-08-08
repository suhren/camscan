import logging

_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_LOG_FMT = "%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s"
_LOG_LEVEL = logging.DEBUG

_formatter = logging.Formatter(fmt=_LOG_FMT, datefmt=_LOG_DATEFMT)

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(_stream_handler)
logger.setLevel(_LOG_LEVEL)
