"""
Main script that performs clustering experiments taking tweets stored on a
database as the main corpus.
"""

import argparse
import logging

from caupo.tags import Tag, get_tags_by_frequency, fetch_tag_from_db

logger = logging.getLogger("caupo")


def cluster_tag(tag: Tag) -> None:
    """
    Given an entity tag, performs clustering and reports result to logs
    # TODO: Store on tags
    """

    # TODO: complete
    print(tag)


def main() -> None:
    """
    Main script that extracts input arguments and runs the script
    """

    parser = argparse.ArgumentParser(description="Performs clustering over tweets stored on the database")
    parser.add_argument("frequency", metavar="FREQ", type="str", choices=["daily", "weekly", "monthly"])
    args = parser.parse_args()

    logger.debug("Getting all tags with `%s` frequency", args.frequency)
    tags = get_tags_by_frequency(args.frequency)

    for tag_name in tags:
        logger.debug("Fetching tag `%s` from database", tag_name)
        tag = fetch_tag_from_db(args.frequency, tag_name)
        cluster_tag(tag)


if __name__ == "__main__":
    main()
