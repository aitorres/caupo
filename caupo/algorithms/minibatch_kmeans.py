"""
Module for testing the scikit-learn Mini Batch K-Means implementation.
"""

from sklearn.cluster import MiniBatchKMeans


def fit(n_clusters, data):
    """
    Runs a dataset over the K-Means algorithm and prints
    and returns the cluster labels.

    Uses k-means++ for initialization of clusters.
    """

    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    mbkmeans.fit(data)

    labels = mbkmeans.labels_
    centers = mbkmeans.cluster_centers_

    return (labels, centers)