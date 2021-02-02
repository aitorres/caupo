import time
import logging

import pymongo

logger = logging.getLogger("caupo")

client = pymongo.MongoClient()
db = client.caupo

class Timer:
    """Context handler to measure time of a function"""

    def __init__(self, action_name):
        self.action_name = action_name
        logger.info("Starting: %s", self.action_name)

        self.start_time = time.time()
        self.end_time = None

    def __enter__(self):
        return self.start_time

    def __exit__(self, type, value, traceback):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

        logger.info("Finishing: %s", self.action_name)
        logger.debug("Done in %s milliseconds.", self.duration)


def get_all_tweets():
    """
    Queries and returns a cursor with all tweets stored in the database.
    """

    return db.tweets.find()


def get_text_from_all_tweets():
    """
    Queries and returns a cursor with the text from all texts (filtering out any
    other attributes)
    """

    return db.tweets.find({}, {"full_text": 1})
