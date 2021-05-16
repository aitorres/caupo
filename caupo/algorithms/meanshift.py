"""
Module for testing the scikit-learn MeanShift algorithm implementation.
"""

from sklearn.cluster import MeanShift


def fit(n_clusters, data):
    """
    Runs a dataset over the K-Means algorithm and prints
    and returns the cluster labels.

    Uses k-means++ for initialization of clusters.
    """

    clustering = MeanShift()
    clustering.fit(data)

    labels = clustering.labels_
    centers = clustering.cluster_centers_

    return (labels, centers)