import logging

# Create logger
logger = logging.getLogger("caupo")
logger.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')

# Add console (standard output) handler to the logger
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)
