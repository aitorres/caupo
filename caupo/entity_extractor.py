"""
Script that collects and performs named entity recognition (NER) on the text of
stored tweets, analyzing the tweets with the required frequency.

The script will check if there's any need to update information, and won't overwrite or
recalculate statistics that would remain the same, in order to reduce runtime and resource
usage.
"""

import argparse
import logging
from datetime import date
from typing import Any, Dict, List, Set

import pymongo

from caupo.preprocessing import map_strange_characters, nlp
from caupo.tags import (Tag, exclude_preexisting_tags,
                        get_collection_by_frequency, get_tags_by_frequency)

# Instantiate logger
logger = logging.getLogger("caupo")


class EntityTag(Tag):
    """
    A wrapper class to hold a collection of extracted entities and metadata about said
    collection.
    """

    def __init__(self, tag: str, frequency: str, dates: List[date]) -> None:
        """Initializes the wrapper"""

        super().__init__(tag, frequency, dates)

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

            # Set new and old hashtags
            tag["hashtags"]["added"] = list(set(tag["hashtags"]["set"]) - set(previous_tag["hashtags"]["set"]))
            tag["hashtags"]["removed"] = list(set(previous_tag["hashtags"]["set"]) - set(tag["hashtags"]["set"]))

            # Update tag on database
            logger.debug("[Tag %s] Replacing with new version", tag["tag"])
            collection.replace_one(
                {'_id': tag['_id']},
                tag,
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
                    "added": [],
                    "removed": [],
                },
                "organizations": {
                    "list": self.organizations,
                    "set": list(self.organizations_set),
                    "unique_amount": len(self.organizations_set),
                    "added": [],
                    "removed": [],
                },
                "locations": {
                    "list": self.locations,
                    "set": list(self.locations_set),
                    "unique_amount": len(self.locations_set),
                    "added": [],
                    "removed": [],
                },
                "persons": {
                    "list": self.persons,
                    "set": list(self.persons_set),
                    "unique_amount": len(self.persons_set),
                    "added": [],
                    "removed": [],
                },
                "misc": {
                    "list": self.misc,
                    "set": list(self.misc_set),
                    "unique_amount": len(self.misc_set),
                    "added": [],
                    "removed": [],
                },
            },
            "hashtags": {
                "list": self.hashtags,
                "set": list(self.hashtags_set),
                "unique_amount": len(self.hashtags_set),
                "added": [],
                "removed": [],
            },
        }

        return json

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
                try:
                    while not (hashtag[-1].isalpha() or hashtag[-1].isdigit()):
                        hashtag = hashtag[:len(hashtag) - 1]
                except IndexError:
                    # We ran out of hashtag, let's skip it then!
                    continue
                self.hashtags.append(hashtag)

    def extract_entities(self) -> None:
        """Extracts and stores several lists of categorized entities within the object's state"""

        if self.all_entities:
            return

        processed_tweets = [nlp(tweet) for tweet in self.tweets]
        # Excluding mentions and/or hashtags from entities
        droppable_entity = lambda X: X.text.startswith("@") or X.text.startswith("#")
        entities_per_tweet = [
            {(map_strange_characters(X.text), X.label_) for X in doc.ents if not droppable_entity(X)}
            for doc in processed_tweets
        ]
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
