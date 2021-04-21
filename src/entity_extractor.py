"""
Script that collects and performs named entity recognition (NER) on the text of
stored tweets, analyzing the tweets with the required frequency.

The script will check if there's any need to update information, and won't overwrite or
recalculate statistics that would remain the same, in order to reduce runtime and resource
usage.
"""

import argparse
import logging
import os
from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Set, Tuple

import bson.regex
import pymongo

import es_core_news_md
from preprocessing import get_stopwords, map_strange_characters
from utils import get_non_unique_content_from_tweets, get_uninteresting_usernames

# Instantiate logger
logger = logging.getLogger("caupo")
logger.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')

# Add console (standard output) handler to the logger
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Creating folder for output
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
BASE_OUTPUT_FOLDER = f"outputs/entity_extractor/{ timestamp }"
os.makedirs(BASE_OUTPUT_FOLDER)

# Add file handler to the logger
file_handler = logging.FileHandler(f'{BASE_OUTPUT_FOLDER}/entity_extractor.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Database settings
client = pymongo.MongoClient('mongodb://127.0.0.1:27019')
db = client.caupo

# Starting point for calculations (inclusive)
INITIAL_DAY_DICT = {
    'daily': date(year=2021, month=1, day=29), # First (full) day registered
    'weekly': date(year=2021, month=2, day=1), # First monday registered
    'monthly': date(year=2021, month=2, day=1), # First day of first month registered
}


class EntityTag:
    """
    A wrapper class to hold a collection of extracted entities and metadata about said
    collection.
    """

    def __init__(self, tag: str, frequency: str, dates: List[date]) -> None:
        """Initializes the wrapper"""

        # Internal settings
        self.tag = tag
        self.frequency = frequency
        self.dates = dates

        # Text to be collected
        self.tweets = None

        # Hashtags to be extracted
        self.hashtags = []

        # Entities to be extracted
        self.all_entities = []
        self.organizations = []
        self.locations = []
        self.persons = []
        self.misc = []

    @staticmethod
    def calculate_related_attributes(frequency) -> None:
        """
        Given a frequency, this static method obtains all the already-stored tags
        by that frequency ordered by tag, and then iteratively calculates variations
        of entities in function of time, and updates each tag
        """

        logger.debug("Calculating related attributes for `%s` tags", frequency)

        # Get all entities by frequency
        collection = get_collection_by_frequency(frequency)
        tags = list(collection.find({'frequency': frequency}).sort([
            ("tag", pymongo.ASCENDING),
        ]))
        logger.debug("Working with %s tags", len(tags))

        # Iterates over fetched tags
        for i, tag in enumerate(tags):
            # We can skip the first one since there's nothing to compare to
            if i == 0:
                logger.debug("[Tag %s] Skipping", tag["tag"])
                continue

            # Set new and old entities
            previous_tag = tags[i - 1]
            entity_types = [
                "all",
                "persons",
                "locations",
                "organizations",
                "misc",
            ]
            for entity_type in entity_types:
                tag["entities"][entity_type]["added"] = list(
                    set(tag["entities"][entity_type]["set"]) - set(previous_tag["entities"][entity_type]["set"])
                )
                tag["entities"][entity_type]["removed"] = list(
                     set(previous_tag["entities"][entity_type]["set"]) - set(tag["entities"][entity_type]["set"])
                )
                logger.debug("[Tag %s] New entities of type %s: %s", tag["tag"], entity_type, tag["entities"][entity_type]["added"])
                logger.debug("[Tag %s] Removed entities of type %s: %s", tag["tag"], entity_type, tag["entities"][entity_type]["removed"])

            # Set new and old hashtags
            tag["hashtags"]["added"] = list(set(tag["hashtags"]["set"]) - set(previous_tag["hashtags"]["set"]))
            tag["hashtags"]["removed"] = list(set(previous_tag["hashtags"]["set"]) - set(tag["hashtags"]["set"]))

            # Update tag on database
            logger.debug("[Tag %s] Replacing with new version", tag["tag"])
            collection.replace_one(
                {'_id': tag['_id']},
                {},
                upsert=False
            )

    @property
    def all_entities_set(self) -> Set[str]:
        """Return a set with all entities currently stored"""

        return set(self.all_entities)

    @property
    def organizations_set(self) -> Set[str]:
        """Return a set with all organization entities currently stored"""

        return set(self.organizations)

    @property
    def locations_set(self) -> Set[str]:
        """Return a set with all location entities currently stored"""

        return set(self.locations)

    @property
    def persons_set(self) -> Set[str]:
        """Return a set with all person entities currently stored"""

        return set(self.persons)

    @property
    def misc_set(self) -> Set[str]:
        """Return a set with all misc entities currently stored"""

        return set(self.misc)

    @property
    def hashtags_set(self) -> Set[str]:
        """Return a set with all hashtags currently stored"""

        return set(self.hashtags)

    @property
    def formatted_dates(self) -> List[str]:
        """Returns a list of the `dates` of this tag in YYYY-MM-DD format"""

        return [date.strftime("%Y-%m-%d") for date in self.dates]


    def to_json(self) -> Dict[str, Any]:
        """Returns a JSON-like representation of the object, ready for storage in MongoDB"""

        json = {
            "tag": self.tag,
            "frequency": self.frequency,
            "first_date": self.dates[0].strftime("%Y-%m-%d"),
            "last_date":  self.dates[-1].strftime("%Y-%m-%d"),
            "dates": self.formatted_dates,
            "dates_amount": len(self.dates),
            "tweets": self.tweets,
            "tweets_amount": len(self.tweets),
            "entities": {
                "all": {
                    "list": self.all_entities,
                    "set": list(self.all_entities_set),
                    "unique_amount": len(self.all_entities_set),
                },
                "organizations": {
                    "list": self.organizations,
                    "set": list(self.organizations_set),
                    "unique_amount": len(self.organizations_set),
                },
                "locations": {
                    "list": self.locations,
                    "set": list(self.locations_set),
                    "unique_amount": len(self.locations_set),
                },
                "persons": {
                    "list": self.persons,
                    "set": list(self.persons_set),
                    "unique_amount": len(self.persons_set),
                },
                "misc": {
                    "list": self.misc,
                    "set": list(self.misc_set),
                    "unique_amount": len(self.misc_set),
                },
            },
            "hashtags": {
                "list": self.hashtags,
                "set": list(self.hashtags_set),
                "unique_amount": len(self.hashtags_set),
            },
        }

        return json


    def load_tweets(self) -> None:
        """Loads and stores a copy of the tweets' text that are covered by this tag"""

        tweets = db.tweets.find(
            {
                "user.screen_name": { "$nin": get_uninteresting_usernames() },
                "full_text": { "$nin": get_non_unique_content_from_tweets() },
                "city_tag": "Caracas",
                "created_at": {
                    "$in": [bson.regex.Regex(f"^{d}") for d in self.formatted_dates]
                }
            },
            { "full_text": 1 }
        )

        self.tweets = list({t["full_text"] for t in tweets})


    def clean_tweets(self) -> None:
        """Cleans certain events from the text of the tweets"""

        if self.tweets is None:
            return

        # Creating different patterns for laughter
        pattern_seeds = ["ja", "JA", "js", "aj", "AJ", "JS", "je", "JE", "ji", "JI"]
        laughter = set()
        for pattern in pattern_seeds:
            laughter = laughter.union({ pattern * i for i in range(2, 6) })

        stopwords = get_stopwords()
        unwanted_words = set(stopwords).union(laughter)

        # Removing laughter in Spanish (jajaja) and stopwords
        for i, tweet in enumerate(self.tweets):
            self.tweets[i] = " ".join(map(lambda x: "" if x in unwanted_words else x, tweet.split()))


    def extract_hashtags(self) -> None:
        """Stores a list of hashtags used in the tweets within the object's state"""

        if self.hashtags:
            return

        for tweet in self.tweets:
            words = tweet.split()
            hashtags = [w for w in words if w.startswith("#")]
            for hashtag in hashtags:
                # Normalizing hashtag
                hashtag = map_strange_characters(hashtag.lower())
                self.hashtags.append(hashtag)


    def extract_entities(self) -> None:
        """Extracts and stores several lists of categorized entities within the object's state"""

        if self.all_entities:
            return

        nlp = es_core_news_md.load()
        processed_tweets = [nlp(tweet) for tweet in self.tweets]
        entities_per_tweet = [{(X.text, X.label_) for X in doc.ents} for doc in processed_tweets]
        entities = set().union(*entities_per_tweet)

        for entity, label in entities:
            self.all_entities.append(entity)

            if label == "PER":
                self.persons.append(entity)
            if label == "ORG":
                self.organizations.append(entity)
            if label == "LOC":
                self.locations.append(entity)
            if label == "MISC":
                self.misc.append(entity)


    def store_in_db(self) -> None:
        """Stores information in database"""

        collection = get_collection_by_frequency(self.frequency)
        object = self.to_json()

        collection.insert_one(object)


    def fetch_and_store(self) -> None:
        """Given a freshly initialized instance, fetches and extracts information and stores result in DB"""

        # Loading info
        logger.debug("[Tag %s] Loading tweets", self.tag)
        self.load_tweets()
        self.clean_tweets()
        logger.debug("[Tag %s] Successfully loaded %s tweets", self.tag, len(self.tweets))

        # Calculations
        logger.debug("[Tag %s] Extracting hashtags", self.tag)
        self.extract_hashtags()
        logger.debug("[Tag %s] Extracting entities", self.tag)
        self.extract_entities()

        logger.debug("[Tag %s] Successfully extracted %s hashtags and %s entities", self.tag,
                     len(self.hashtags), len(self.all_entities))

        # Store
        logger.debug("[Tag %s] Storing information...", self.tag)
        self.store_in_db()
        logger.debug("[Tag %s] Information stored", self.tag)


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
            for month in range (1, 12 + 1):
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


def get_collection_by_frequency(frequency: str) -> pymongo.collection.Collection:
    """Given a frequency, returns the appropriate collection where information should be stored"""

    collection_name = f"entities_{frequency}"
    collection = getattr(db, collection_name)
    return collection


def get_args_parser() -> argparse.ArgumentParser:
    """Builds a return a parser with required arguments for this script"""

    parser = argparse.ArgumentParser(description='Extracts and storesnamed entities from tweets in a given frequency.')
    parser.add_argument('frequency', metavar='FREQ', type=str,
                        choices=['daily', 'weekly', 'monthly'],
                        help='Frequency for query (supported values: daily, weekly, monthly)')
    parser.add_argument('--recalculate', action='store_true',
                        help='Force recalculation of already stored values')
    return parser


def main() -> None:
    """
    Main function that executes the Entity Extractor script.
    Reads arguments and flags from standard input, initializes required variables
    and calculates and stores named entities in the database
    """

    # Parse arguments
    parser = get_args_parser()
    args = parser.parse_args()
    frequency = args.frequency
    recalculate = args.recalculate

    logger.debug("Main execution started")

    # Get tags for requested frequency
    tags = get_tags_by_frequency(frequency)
    logger.info("Amount of tags: %s", len(tags))

    # Unless required to recalculate, drop tags that have already been stored
    if not recalculate:
        tags = exclude_preexisting_tags(frequency, tags)
    else:
        logger.info("Including all tags")

    # Initializing an entity tag instance for each tag
    entity_tags = [EntityTag(name, frequency, dates) for name, dates in tags]

    # Storing individual tags
    for entity_tag in entity_tags:
        entity_tag.fetch_and_store()

    # Updating tags with variations
    EntityTag.calculate_related_attributes(frequency)

    logger.debug("Main execution finished")


# For running the main script
if __name__ == "__main__":
    main()
