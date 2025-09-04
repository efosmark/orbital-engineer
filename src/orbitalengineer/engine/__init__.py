import logging
import os

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("orbit.engine")
from logging.handlers import RotatingFileHandler

file_handler = RotatingFileHandler("/tmp/debug.log", maxBytes=5_000_000, backupCount=3)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(file_handler)

log_level_env:str|None = os.environ.get("LOG_LEVEL", None)
if log_level_env:
    levels_by_name = logging.getLevelNamesMapping()
    level = levels_by_name[log_level_env.upper()]
    
    logger.setLevel(level)
    logger.propagate = False
    handler = logging.StreamHandler()

    # New formatter
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

else:
    logger.setLevel(logging.WARN)