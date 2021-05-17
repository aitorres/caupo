import logging
import os

# Create logger
logger = logging.getLogger('backend')
log_level = getattr(logging, os.environ.get('LOG_LEVEL', 'DEBUG'))
logger.setLevel(log_level)

# Create formatter
formatter = logging.Formatter(fmt="%(asctime)s [%(process)d] [%(levelname)s] %(message)s",
                              datefmt="[%Y-%m-%d %H:%M:%S %z]")

# Add console (standard output) handler to the logger
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)

# If set, adds file handler to the logger
log_file_path = os.environ.get('LOG_FILE_PATH')
if log_file_path:
    file_handler = logging.FileHandler(log_file_path)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(file_handler)
