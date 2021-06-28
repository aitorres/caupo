"""
Module to hold endpoints that will serve information about clustering and cluster results
extracted from the dataset of tweets periodically.

Connected to the main Flask application through a Blueprint.
"""

import json
import logging
from typing import Any, Dict, Tuple

import pandas as pd
from flask import Blueprint, request

from caupo.clustering import get_clustering_functions
from caupo.database import get_results_collection
from caupo.embeddings import get_embedder_function_names
from caupo.results import calculate_consolidated_data

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
    }
    result = RESULT_COLLECTION.find_one(query_filter, {'scores': 1})

    if not result:
        sil_score = None
    else:
        sil_score = result["scores"].get("silhouette")

    return {
        'httpStatus': 200,
        'message': "Result fetched successfully.",
        'data': sil_score,
    }, 200


@blueprint.route('/results/list', methods=["POST"])
def get_valid_results() -> Tuple[Dict[str, Any], int]:
    """
    Returns all valid silhouette scores for a valid result (combinaiton of frequency and tag)
    """

    data = request.json
    frequency = data.get('frequency')
    tag = data.get('tag')
    if None in [frequency, tag]:
        return {
            'httpStatus': 400,
            'message': "Invalid params!",
            'data': None,
        }, 400

    logger.info(
        "Requested silhouette score with freq `%s`, tag `%s`",
        frequency, tag)

    query_filter = {
        'frequency': frequency,
        'tag': tag,
        'success': True,
    }
    results = list(RESULT_COLLECTION.find(
        query_filter,
        {
            '_id': 0,
            'frequency': 1,
            'tweetsAmount': 1,
            'validTweetsAmount': 1,
            'minClusterSize': 1,
            'maxClusterSize': 1,
            'avgClusterSize': 1,
            'tag': 1,
            'algorithm': 1,
            'embedder': 1,
            'clusterThemes': 1,
            'time': 1,
            'averageSentiment': 1,
            'scores': 1
        }
    ))

    valid_results = [r for r in results if r["scores"]["silhouette"] is not None]
    sorted_results = sorted(valid_results, key=lambda r: r["scores"]["silhouette"], reverse=True)

    return {
        'httpStatus': 200,
        'message': "Results fetched successfully.",
        'data': sorted_results,
    }, 200


@blueprint.route('/results/consolidated/<frequency>/', methods=['GET'])
def get_consolidated_results(frequency: str) -> Tuple[Dict[str, Any], int]:
    """
    Returns consolidated, aggregated results
    """

    file_path = f"/root/tesis/outputs/cluster_tags/{frequency}/results.csv"
    data = pd.read_csv(file_path)

    try:
        consolidated_results = calculate_consolidated_data(frequency, data)
    except AssertionError:
        return {
            'httpStatus': 400,
            'message': "Invalid params!",
            'data': None,
        }, 400

    consolidated_json = json.loads(
        consolidated_results.to_json(
            orient='index',
        )
    )
    response_json = [
        {
            **data,
            'algorithm': key[0],
            'embedder': key[1]
        }
        for key, data in consolidated_json.items()
    ]

    return {
        'httpStatus': 200,
        'message': "Aggregated data returned successfully!",
        'data': response_json,
    }
