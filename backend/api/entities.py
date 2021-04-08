"""
Module to hold endpoints that will serve information about (named) entities extracted
from the dataset of tweets periodically.

Connected to the main Flask application through a Blueprint.
"""

import logging
from typing import Any, Dict, List, Tuple

import pymongo
from flask import Blueprint

# Initializing logger
logger = logging.getLogger()

# Initializing connection to the database
client = pymongo.MongoClient('mongodb://127.0.0.1:27019')
db = client.caupo

# Initializing Flask blueprint to register endpoints with main app
blueprint = Blueprint('entities', __name__, url_prefix='/entities')

# For validation
VALID_FREQUENCIES = [
    'daily',
    'weekly',
    'monthly',
]


# Endpoints
@blueprint.route('/get/<frequency>', methods=['GET'])
def get_entities(frequency: str) -> Tuple[Dict[str, Any], int]:
    """
    Given a frequency, returns a list with the information of all entities
    stored within tags for that frequency
    """

    if frequency not in VALID_FREQUENCIES:
        logger.error("[get_entities] Tried to get entities with invalid frequency `%s`", frequency)
        return {
            'httpStatus': 400,
            'message': f'You requested an invalid or unrecognized frequency (`{frequency}`)'
        }, 400

    entities = _fetch_entities(frequency)
    return {
        'httpStatus': 200,
        'message': 'Entities collected successfully',
        'data': entities
    }, 200


# Auxiliary functions
def _fetch_entities(frequency: str) -> List[Dict[str, Any]]:
    """
    Given a value for frequency, fetches and returns the information for
    entities of that type stored in the database
    """

    if frequency not in VALID_FREQUENCIES:
        logger.warning("[_fetch_entities] Tried to fetch entities with invalid frequency `%s`", frequency)
        return []

    collection = _get_collection_by_frequency(frequency)
    entities = collection.find({}, {
        "_id": False,
        "tag": True,
        "frequency": True,
        "tweets_amount": True,
        "entities": True,
        "hashtags": True,
    })
    return list(entities)


# TODO: Unify with similar function in main `src` of project
def _get_collection_by_frequency(frequency: str) -> pymongo.collection.Collection:
    """Given a frequency, returns the appropriate collection where information should be stored"""

    collection_name = f"entities_{frequency}"
    collection = getattr(db, collection_name)
    return collection
