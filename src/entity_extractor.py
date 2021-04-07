"""
Script that collects and performs named entity recognition (NER) on the text of
stored tweets, analyzing the tweets with the required frequency.

The script will check if there's any need to update information, and won't overwrite or
recalculate statistics that would remain the same, in order to reduce runtime and resource
usage.
"""

import argparse
from calendar import monthrange
from datetime import date, timedelta
from typing import List, Tuple

import pymongo

# Database settings
client = pymongo.MongoClient('mongodb://127.0.0.1:27019')
db = client.caupo

# Starting point for calculations (inclusive)
INITIAL_DAY_DICT = {
    'daily': date(year=2021, month=1, day=29), # First (full) day registered
    'weekly': date(year=2021, month=2, day=1), # First monday registered
    'monthly': date(year=2021, month=2, day=1), # First day of first month registered
}


class EntityListWrapper:
    """
    A wrapper class to hold a collection of extracted entities and metadata about said
    collection.
    """

    def __init__(self, frequency: str, tweets: List[str]) -> None:
        """Initializes the wrapper"""

        # Internal settings
        self.frequency = frequency
        self.tag = tag
        self.tweets = tweets

        # Hashtags to be extracted
        self.hashtags = None

        # Entities to be extracted
        self.all_entities = None
        self.organizations = None
        self.locations = None
        self.persons = None
        self.misc = None


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
        weeks = (most_recent_monday - initial_day).days / 7

        tags = []
        for week_number in range(weeks):
            week_monday = today + timedelta(days=7 * week_number)

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


def get_collection_by_frequency(frequency: str) -> pymongo.collection.Collection:
    """Given a frequency, returns the appropriate collection where information should be stored"""

    collection_name = f"entities_{frequency}"
    collection = getattr(client, collection_name)
    return collection


def get_args_parser() -> argparse.ArgumentParser:
    """Builds a return a parser with required arguments for this script"""

    parser = argparse.ArgumentParser(description='Extracts and storesnamed entities from tweets in a given frequency.')
    parser.add_argument('frequency', metavar='FREQ', type=str,
                        choices=['daily', 'weekly', 'monthly'],
                        help='Frequency for query (supported values: daily, weekly, monthly)')
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

    raise NotImplementedError("Main script not implemented")

# For running the main script
if __name__ == "__main__":
    main()
