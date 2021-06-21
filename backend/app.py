"""
Main back-end module for CAUPO HTTP API that will be used for data and cluster
visualization in real-time.

Defines a Flask web application that can be used with gunicorn or any other server that
supports WSGI applications.
"""

import logging
from typing import Any, Dict, Tuple

from flask import Flask, request
from flask_cors import CORS

from backend.api import clusters, entities

# Logger initialization
logger = logging.getLogger('backend')

# Flask app initialization
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)


# Auxiliary endpoints
@app.route('/ping', methods=['GET'])
def ping() -> Tuple[Dict[str, Any], int]:
    """Auxiliary endpoint to be used for health checks from external applications"""

    return {
        'httpStatus': 200,
        'message': "pong"
    }, 200


# Error handlers
@app.errorhandler(404)
def not_found_error_handler(error: Any) -> Tuple[Dict[str, Any], int]:
    """
    Handler that returns a 404 error in a standard format.

    :param error: error context
    :return: HTTP Response tuple for a Not Found error
    """

    # Logging error
    url = request.url
    if "favicon.ico" not in url:
        logger.warning("Request for URL `%s` not found: %s", url, error)

    return {
        "errorType": "NotFound",
        "httpStatus": 404,
        "message": "Page not found!"}, 404


@app.errorhandler(500)
def internal_server_error_handler(error: Any) -> Tuple[Dict[str, Any], int]:
    """
    Handler that returns an internal server error information in a standard form
at.

    :param error: exception information received from Flask
    :return: HTTP Response tuple for an Internal Server error
    """

    # Logging received error
    logger.error("Internal server error %s", error, exc_info=True)

    return {
        "errorType": "InternalServerError",
        "httpStatus": 500,
        "message": str(error)}, 500


# Blueprints for functional endpoints
app.register_blueprint(entities.blueprint)
app.register_blueprint(clusters.blueprint)


# Runs the Flask application server
if __name__ == '__main__':
    app.run()
