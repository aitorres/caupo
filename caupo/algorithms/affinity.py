"""
Module for testing the scikit-learn Affinity Propagation algorithm implementation.
"""

from sklearn.cluster import AffinityPropagation


def fit(n_clusters, data):
    """
    Runs a dataset over the K-Means algorithm and prints
    and returns the cluster labels.

    Uses k-means++ for initialization of clusters.
    """

    clustering = AffinityPropagation()
    clustering.fit(data)

    labels = clustering.labels_
    centers = clustering.cluster_centers_

    return (labels, centers)