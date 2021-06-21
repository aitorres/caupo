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
