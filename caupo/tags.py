"""
Auxiliary module to provide implementation of a class and several functions to
store tagged groups of tweets and related contend, analysis results and other
associated metadata, and to be crossimported in other scripts.

Each Tag represents one cohesive unit of content at a given time window, specified
by its frequency (daily, weekly or monthly).
"""

import logging
from calendar import monthrange
from datetime import date, timedelta
from typing import List, Tuple

import bson.regex
import pymongo

from caupo.preprocessing import get_stopwords
from caupo.utils import (get_non_unique_content_from_tweets,
                         get_uninteresting_usernames)

# Starting point for calculations (inclusive)
INITIAL_DAY_DICT = {
    'daily': date(year=2021, month=1, day=29),  # First (full) day registered
    'weekly': date(year=2021, month=2, day=1),  # First monday registered
    'monthly': date(year=2021, month=2, day=1),  # First day of first month registered
}

# Instantiate logger
logger = logging.getLogger("caupo")

# Database settings
# TODO: Unify and import everywhere from `database` module
client = pymongo.MongoClient('mongodb://127.0.0.1:27019')
db = client.caupo


class Tag:
    """
    A wrapper class that holds temporarily-related tweets, content and other metadata
    over a cohesive temporal dimension.

    Each Tag instance represents one cohesive unit of content at a given time window,
    specified by its frequency (daily, weekly or monthly).
    """

    def __init__(self, tag: str, frequency: str, dates: List[date]) -> None:
        """Initializes the wrapper"""

        # Internal settings
        self.tag = tag
        self.frequency = frequency
        self.dates = dates

        # Text to be collected
        self.tweets = None

    def load_tweets(self) -> None:
        """Loads and stores a copy of the tweets' text that are covered by this tag"""

        tweets = db.tweets.find(
            {
                "user.screen_name": {"$nin": get_uninteresting_usernames()},
                "full_text": {"$nin": get_non_unique_content_from_tweets()},
                "city_tag": "Caracas",
                "created_at": {
                    "$in": [bson.regex.Regex(f"^{d}") for d in self.formatted_dates]
                }
            },
            {"full_text": 1}
        )

        self.tweets = list({t["full_text"] for t in tweets})

    def clean_tweets(self) -> None:
        """Cleans certain events from the text of the tweets"""

        if self.tweets is None:
            return

        # Creating different patterns for laughter
        pattern_seeds = ["ja", "JA", "js", "aj", "AJ", "JS", "je", "JE", "ji", "JI", "Jajajaja", "Jajaja"]
        laughter = set()
        for pattern in pattern_seeds:
            laughter = laughter.union({pattern * i for i in range(1, 6)})

        stopwords = get_stopwords()
        unwanted_words = set(stopwords).union(laughter)

        # Removing laughter in Spanish (jajaja) and stopwords
        for i, tweet in enumerate(self.tweets):
            self.tweets[i] = " ".join(map(lambda x: "" if x in unwanted_words else x, tweet.split()))


def get_collection_by_frequency(frequency: str) -> pymongo.collection.Collection:
    """Given a frequency, returns the appropriate collection where information should be stored"""

    collection_name = f"entities_{frequency}"
    collection = getattr(db, collection_name)
    return collection


def get_tags_by_frequency(frequency: str) -> List[Tuple[str, List[date]]]:
    """
    Given a frequency, returns a list of 2-tuples (pairs) that contain a tag on the first position
    and a list of days to consider in the second position.

    Valid tags depend on the frequency, e.g. the date for `daily`, name + year of month for `monthly`,
    and range of days for `weekly`.
    """

    initial_day = INITIAL_DAY_DICT[frequency]
    today = date.today()

    if frequency == 'daily':
        # Calculates distance in days (amount of days to consider)
        distance_in_days = (today - initial_day).days

        # Gets a list of all days between initial day and *yesterday* (since today is still on-going)
        days = [initial_day + timedelta(days=i) for i in range(distance_in_days)]
        return [(day.strftime("%Y-%m-%d"), [day]) for day in days]

    if frequency == 'weekly':
        # Calculate distance in weeks
        most_recent_monday = today - timedelta(today.weekday())
        weeks = (most_recent_monday - initial_day).days // 7

        tags = []
        for week_number in range(weeks):
            week_monday = initial_day + timedelta(days=7 * week_number)

            # Gets list of all days in this week, from monday to sunday
            days = [week_monday + timedelta(days=i) for i in range(7)]
            tag_start = week_monday.strftime("%Y-%m-%d")
            tag_end = days[-1].strftime("%Y-%m-%d")
            week_tag = f"{tag_start} - {tag_end}"
            tags.append((week_tag, days))
        return tags

    if frequency == 'monthly':
        initial_year, initial_month = initial_day.year, initial_day.month
        today_year, today_month = today.year, today.month

        tags = []
        for year in range(initial_year, today_year + 1):
            for month in range(1, 12 + 1):
                if year == initial_year and month < initial_month:
                    continue

                if year == today_year and month >= today_month:
                    break

                month_tag = f"{year}-{str(month).zfill(2)}"

                num_days = monthrange(year, month)[1]
                days = [date(year, month, day) for day in range(1, num_days + 1)]
                tags.append((month_tag, days))
        return tags

    raise NotImplementedError(f"Value of `frequency` = `{frequency}` not supported in `get_tags_by_frequency`")


def exclude_preexisting_tags(frequency: str, tags: List[Tuple[str, List[date]]]) -> List[Tuple[str, List[date]]]:
    """
    Given a frequency and a list of tags and dates, returns a new list containing only tags
    that don't already exist on the database.
    """

    collection = get_collection_by_frequency(frequency)
    filtered_tags = []

    for tag in tags:
        exists = collection.count_documents({"tag": tag[0]}) > 0

        if not exists:
            filtered_tags.append(tag)
        else:
            logger.info("Excluding tag %s", tag[0])

    return filtered_tags
