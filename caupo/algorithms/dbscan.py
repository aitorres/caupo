"""
Module for testing the scikit-learn DBSCAN algorithm implementation.
"""

from sklearn.cluster import DBSCAN


def fit(n_clusters, data):
    """
    Runs a dataset over the K-Means algorithm and prints
    and returns the cluster labels.

    Uses k-means++ for initialization of clusters.
    """

    clustering = DBSCAN()
    clustering.fit(data)

    labels = clustering.labels_
    try:
        centers = clustering.cluster_centers_
    except:
        centers = None

    return (labels, centers)