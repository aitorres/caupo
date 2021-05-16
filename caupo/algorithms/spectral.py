"""
Module for testing the scikit-learn Spectral Clustering algorithm implementation.
"""

from sklearn.cluster import SpectralClustering


def fit(n_clusters, data):
    """
    Runs a dataset over the K-Means algorithm and prints
    and returns the cluster labels.

    Uses k-means++ for initialization of clusters.
    """

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        random_state=0,
        assign_labels="discretize"
    )
    clustering.fit(data)

    labels = clustering.labels_
    try:
        centers = clustering.cluster_centers_
    except:
        centers = None

    return (labels, centers)