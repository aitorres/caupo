import time
import logging

import emoji

from pymongo import MongoClient

logger = logging.getLogger("caupo")

client = MongoClient('mongodb://127.0.0.1:27019')
db = client.caupo

class Timer:
    """Context handler to measure time of a function"""

    def __init__(self, action_name):
        self.action_name = action_name
        logger.info("Starting: %s", self.action_name)

        self.start_time = time.time()
        self.end_time = None
        self.duration = None

    def __enter__(self):
        return self.start_time

    def __exit__(self, type, value, traceback):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

        logger.info("Finishing: %s", self.action_name)
        logger.debug("Done in %.2f seconds.", self.duration)


def get_all_tweets():
    """
    Queries and returns a cursor with all tweets stored in the database.
    """

    return db.tweets.find()


def get_non_unique_tweets():
    """
    Queries and returns a cursor with all the tweets which content, taken as the `full_text`
    of the document structure, is repeated verbatim in another tweet.
    """

    return db.tweets.aggregate(
        [
            {
                "$group": {
                    "_id": "$full_text",
                    "count": {
                        "$sum": 1
                    }
                }
            },
            {
                "$match": {
                    "count": {
                        "$gt": 1
                    }
                }
            }
        ]
    )


def get_non_unique_content_from_tweets():
    """Returns a list with the text of non-unique tweets."""

    return [doc['_id'] for doc in get_non_unique_tweets()]


def get_text_from_all_tweets(exclude_uninteresting_usernames=True, exclude_uninteresting_text=True):
    """
    Queries and returns a cursor with the text from all texts (filtering out any
    other attributes)
    """

    if exclude_uninteresting_usernames:
        uninteresting_usernames = [
            "SismosVenezuela",
            "DolarBeta",
            "tiempo_caracas",
        ]
    else:
        uninteresting_usernames = []

    if exclude_uninteresting_text:
        uninteresting_text = get_non_unique_content_from_tweets()
    else:
        uninteresting_text = []

    return db.tweets.find(
        {
            "user.screen_name": { "$nin": uninteresting_usernames },
            "full_text": { "$nin": uninteresting_text },
        },
        {
            "full_text": 1
        }
    )


def remove_emoji(phrase):
    """Removes all emojis from a phrase"""

    return emoji.get_emoji_regexp().sub(r'', phrase)


def remove_accents(phrase):
    """Removes all accents (áéíóú) from a lowercase phrase"""

    accents_map = {
        'á': 'a',
        'é': 'e',
        'í': 'i',
        'ó': 'o',
        'ú': 'u',
    }

    return "".join(map(lambda x: x if x not in accents_map else accents_map[x], phrase))
