"""
Module to hold endpoints that will serve information about clustering and cluster results
extracted from the dataset of tweets periodically.

Connected to the main Flask application through a Blueprint.
"""

import logging
from typing import Any, Dict, Tuple

from flask import Blueprint

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
    }


@blueprint.route('/tags/list/<frequency>', methods=['GET'])
def get_valid_tags_list(frequency: str) -> Tuple[Dict[str, Any], int]:
    """
    Returns the name (tag name) of all valid tags for the given frequency
    """

    stored_tags = list(RESULT_COLLECTION.find({'frequency': frequency}, {'tag': 1}))

    return {
        'httpStatus': 200,
        'message': 'List of valid tag names retrieved successfully',
        'data': [tag['tag'] for tag in stored_tags]
    }


@blueprint.route('/results/count/<frequency>/<algorithm>/<embedder>/', methods=["GET"])
def get_result_count(frequency: str, algorithm: str, embedder: str) -> Tuple[Dict[str, Any], int]:
    """
    Returns the count of all valid results from a combination of embedder model and algorithm.
    """

    query_filter = {
        'frequency': frequency,
        'embedder': embedder,
        'algorithm': algorithm,
        'success': True,  # excludes results that failed before returning a labelling of data
        'sil_score': {"$ne": None}  # excludes results that returned less than 2 valid clusters
    }

    results_count = RESULT_COLLECTION.count_documents(query_filter)
    return {
        'httpStatus': 200,
        'message': "Results counted successfully.",
        'data': {
            'count': results_count
        }
    }
