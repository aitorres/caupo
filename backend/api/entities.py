"""
Module to hold endpoints that will serve information about (named) entities extracted
from the dataset of tweets periodically.

Connected to the main Flask application through a Blueprint.
"""

import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Tuple, Union

import pymongo
from flask import Blueprint
from wordcloud import WordCloud

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
@blueprint.route('/wordcloud/get/<frequency>', methods=['GET'])
def get_wordcloud(frequency: str) -> Tuple[Dict[str, Any], int]:
    """
    Given a frequency, returns the base64-encoded data of an image that represents the
    wordcloud of all entities found in the latest tag of said frequency
    """

    if frequency not in VALID_FREQUENCIES:
        logger.error("[get_entities] Tried to get entities with invalid frequency `%s`", frequency)
        return {
            'httpStatus': 400,
            'message': f'You requested an invalid or unrecognized frequency (`{frequency}`)'
        }, 400

    # Getting latest entity tag
    entity = _fetch_entities(frequency, 1)[0]
    image_data = _get_b64_wordcloud(entity)

    return {
        'httpStatus': 200,
        'message': 'Wordcloud generated collected successfully',
        'data': image_data
    }, 200


@blueprint.route('/variations/get/<frequency>/', methods=['GET'])
@blueprint.route('/variations/get/<frequency>/<tag>/', methods=['GET'])
def get_variation_data(frequency: str, tag: str = "") -> Tuple[Dict[str, Any], int]:
    """
    Given a frequency, obtains the informations of entity variations (added and removed entities)
    for all tags and returns it. If a given tag is specified, returns only the information for
    said tag
    """

    if frequency not in VALID_FREQUENCIES:
        logger.error("[get_entities] Tried to get entities with invalid frequency `%s`", frequency)
        return {
            'httpStatus': 400,
            'message': f'You requested an invalid or unrecognized frequency (`{frequency}`)'
        }, 400

    # Fetch entities from DB
    entities = _fetch_entities(frequency)

    # Preprocess entities
    filtered_tags = []
    for entity_tag in entities:
        filtered_tag = {
            'tag': entity_tag['tag'],
            'frequency': frequency,
            'tweets_amount': entity_tag['tweets_amount'],
        }
        for entity_type in ["all", "misc", "persons", "locations", "organizations"]:
            filtered_tag[entity_type]["added"] = entity_tag["entities"][entity_type]["added"]
            filtered_tag[entity_type]["removed"] = entity_tag["entities"][entity_type]["removed"]
        filtered_tag["hashtags"]["added"] = entity_tag["hashtags"]["added"]
        filtered_tag["hashtags"]["removed"] = entity_tag["hashtags"]["removed"]
        filtered_tags.append(filtered_tag)

    # If a specific tag was requested, keep just that one
    if tag:
        filtered_tags = [ft for ft in filtered_tags if ft['tag'] == tag]

    return {
        'httpStatus': 200,
        'message': 'Entity variation data collected successfully',
        'data': filtered_tags
    }, 200


@blueprint.route('/get/<frequency>', methods=['GET'])
@blueprint.route('/get/<frequency>/<amount>', methods=['GET'])
def get_entities(frequency: str, amount: Union[str, int] = 0) -> Tuple[Dict[str, Any], int]:
    """
    Given a frequency, returns a list with the information of all entities
    stored within tags for that frequency
    """

    if isinstance(amount, str):
        if amount.isdigit():
            amount = int(amount)
        else:
            logger.error("[get_entities] Tried to get entities with invalid amount `%s`", amount)
            return {
                'httpStatus': 400,
                'message': f'You requested an invalid or unrecognized amount (`{amount}`)'
            }, 400

    if frequency not in VALID_FREQUENCIES:
        logger.error("[get_entities] Tried to get entities with invalid frequency `%s`", frequency)
        return {
            'httpStatus': 400,
            'message': f'You requested an invalid or unrecognized frequency (`{frequency}`)'
        }, 400

    entities = _fetch_entities(frequency, amount)
    return {
        'httpStatus': 200,
        'message': 'Entities collected successfully',
        'data': entities
    }, 200


# Auxiliary functions
def _fetch_entities(frequency: str, amount: int = 0) -> List[Dict[str, Any]]:
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
    }).sort([
        ("tag", pymongo.DESCENDING),
    ])
    if amount > 0:
        entities = entities.limit(amount)
    return list(entities)


def _get_b64_wordcloud(entity) -> str:
    """
    Given an entity tag, generates a wordcloud from all of its entities and returns
    a base64 encoded string representing said wordcloud
    """

    wc = WordCloud(width=800, height=400, min_word_length=3, max_words=150).generate(" ".join(entity['entities']['persons']['list']))
    buffer = BytesIO()
    wc.to_image().save(buffer, 'png')
    b64 = base64.b64encode(buffer.getvalue())

    return b64.decode('ascii')


# TODO: Unify with similar function in main `src` of project
def _get_collection_by_frequency(frequency: str) -> pymongo.collection.Collection:
    """Given a frequency, returns the appropriate collection where information should be stored"""

    collection_name = f"entities_{frequency}"
    collection = getattr(db, collection_name)
    return collection
