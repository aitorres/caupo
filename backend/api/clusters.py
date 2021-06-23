"""
Module to hold endpoints that will serve information about clustering and cluster results
extracted from the dataset of tweets periodically.

Connected to the main Flask application through a Blueprint.
"""

import logging
from typing import Any, Dict, Tuple

from flask import Blueprint, request

from caupo.database import get_results_collection
from caupo.clustering import get_clustering_functions
from caupo.embeddings import get_embedder_function_names

# Initializing logger
logger = logging.getLogger('backend')

# Initializing Flask blueprint to register endpoints with main app
blueprint = Blueprint('clusters', __name__, url_prefix='/clusters')

# Preloading data
EMBEDDER_FUNCTION_NAMES = get_embedder_function_names()
CLUSTERING_ALGORITHM_NAMES = list(get_clustering_functions().keys())
RESULT_COLLECTION = get_results_collection()


# Endpoints
@blueprint.route('/embedders/list', methods=["GET"])
def get_embedder_names() -> Tuple[Dict[str, Any], int]:
    """
    Returns the name of all valid embedders as used in the main `caupo` module.
    """

    return {
        'httpStatus': 200,
        'message': "List of embedders retrieved successfully.",
        'data': EMBEDDER_FUNCTION_NAMES,
    }


@blueprint.route('/clustering-algorithms/list', methods=["GET"])
def get_clustering_algorithm_list() -> Tuple[Dict[str, Any], int]:
    """
    Returns the name of all valid clustering algorithms as used in the main `caupo` module.
    """

    return {
        'httpStatus': 200,
        'message': "List of clustering algorithms retrieved successfully.",
        'data': CLUSTERING_ALGORITHM_NAMES,
    }, 200


@blueprint.route('/tags/list/<frequency>', methods=['GET'])
def get_valid_tags_list(frequency: str) -> Tuple[Dict[str, Any], int]:
    """
    Returns the name (tag name) of all valid tags for the given frequency
    """

    stored_tags = list(RESULT_COLLECTION.find({'frequency': frequency}, {'tag': 1}))

    return {
        'httpStatus': 200,
        'message': 'List of valid tag names retrieved successfully',
        'data': sorted({tag['tag'] for tag in stored_tags}, reverse=True)
    }, 200


@blueprint.route('/results/silhouette', methods=["POST"])
def get_silhouette_score() -> Tuple[Dict[str, Any], int]:
    """
    Returns the silhouette score for a vlid result (combinaiton of frequency, tag, algorithm and embedder)
    """

    data = request.json
    frequency = data.get('frequency')
    embedder = data.get('embedder')
    algorithm = data.get('algorithm')
    tag = data.get('tag')
    if None in [frequency, embedder, algorithm, tag]:
        return {
            'httpStatus': 400,
            'message': "Invalid params!",
            'data': None,
        }, 400

    logger.info(
        "Requested silhouette score with freq `%s`, embedder `%s`, algorithm `%s`, tag `%s`",
        frequency, embedder, algorithm, tag)

    query_filter = {
        'frequency': frequency,
        'embedder': embedder,
        'algorithm': algorithm,
        'tag': tag,
        'success': True,
        'sil_score': {"$ne": None},
    }
    result = RESULT_COLLECTION.find_one(query_filter, {'sil_score': 1})

    if not result:
        sil_score = None
    else:
        sil_score = result["sil_score"]

    return {
        'httpStatus': 200,
        'message': "Result fetched successfully.",
        'data': sil_score,
    }, 200
