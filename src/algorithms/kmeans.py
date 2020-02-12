"""
Module for testing the scikit-learn K-Means implementation.
"""

from sklearn.cluster import KMeans

def fit(n_clusters, data):
    """
    Runs a dataset over the K-Means algorithm and prints
    and returns the cluster labels.

    Uses k-means++ for initialization of clusters.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    return (labels, centers)