"""
Auxiliary module to add wrappers around clustering algorithms
in order to reuse them for different tasks
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import (DBSCAN, OPTICS, AffinityPropagation, KMeans, MeanShift,
                             SpectralClustering)
from sklearn.metrics import silhouette_score

logger = logging.getLogger("caupo")


class BaseClustering:
    """
    Base class to wrap functionality of clustering algorithms
    """

    MIN_CLUSTER_SIZE = 4
    RANDOM_SEED = 12345678

    def __init__(self) -> None:
        """Initializes a new instance of the clustering algorithm"""

        raise NotImplementedError("__init__() not implemented")

    def cluster(self, vectors: List[List[float]]) -> List[int]:
        """Given a list of vectors, performs clustering and returns labels of the output"""

        raise NotImplementedError("cluster() not implemented")

    def remove_small_clusters(self, labels: List[int]) -> List[int]:
        """Given a labeling result, transforms small clusters into noise"""

        # Determining noise labels
        noise_labels = set()
        for label in set(labels):
            if label == -1 or labels.count(label) < self.MIN_CLUSTER_SIZE:
                noise_labels.add(label)

        # `label_map` will help transform the non-continuous non-noise label set to a continuous sequence
        label_map, next_label = {}, 0
        new_labels = []
        for label in labels:
            if label not in label_map.keys():
                if label in noise_labels:
                    # noise is always -1
                    label_map[label] = -1
                else:
                    # we start from 0 and map old labels to a new linear sequence
                    label_map[label] = next_label
                    next_label += 1
            new_labels.append(label_map[label])

        return new_labels


class KMeansClustering(BaseClustering):
    """
    KMeans Clustering wrapper class. Performs clustering using sklearns' implementation
    of the KMeans algorithm. If not specified, automatically detects the proper value of
    K that fits the algorithm best.
    """

    MIN_K = 2
    MAX_K = 5

    def __init__(self, k: Optional[int] = None) -> None:
        """Initializes a new instance of the KMeans clustering algorithm"""

        if k is not None:
            if not self.MIN_K <= k <= self.MAX_K:
                raise ValueError(k)

            self.k = k
            self.model = self.instantiate_model(self.k)
            logger.debug("Initializing KMeansClustering with k=`%s`", self.k)
        else:
            self.k = None
            logger.debug("Deferring KMeans model initialization until vectors are known")

    def instantiate_model(self, k: int) -> KMeans:
        """Instantiates Sklearn's KMeans class and stores within wrapper class"""

        return KMeans(n_clusters=k, random_state=self.RANDOM_SEED)

    def cluster(self, vectors: List[List[float]]) -> List[int]:
        """Given a list of vectors, performs kmeans based clustering and returns labels of the output"""

        # Easy case: we know `k` beforehand
        if self.k is not None:
            return self.model.fit_predict(vectors)

        # Determining best K
        sil_scores = {}
        results = {}
        for k in range(self.MIN_K, self.MAX_K + 1):
            logger.debug("Trying k=`%s`...", k)
            model = self.instantiate_model(k)
            output = model.fit_predict(vectors)
            sil_score = silhouette_score(vectors, output)
            sil_scores[k] = sil_score
            results[k] = output

        # Settings best K
        max_sil = sorted(sil_scores.items(), key=lambda x: x[1], reverse=True)[0]
        self.k = max_sil[0]
        self.model = self.instantiate_model(self.k)
        logger.debug("Maximum silhouette score achieved with k=%s (silhouette score: %s)", max_sil[0], max_sil[1])

        # Get and return the real result
        return results[self.k]


class KMeansNoNoiseClustering(KMeansClustering):
    """
    KMeans Clustering wrapper class, with a basic procedure for removing noise points e.g. points
    that are clustered by themselves and that may be labelled as noise.
    """

    def cluster(self, vectors: List[List[float]]) -> List[int]:
        # First round of clustering:
        initial_labels = super().cluster(vectors)
        if isinstance(initial_labels, np.ndarray):
            initial_labels = initial_labels.tolist()

        # Checking if we have noise
        label_sizes = {label: initial_labels.count(label) for label in set(initial_labels)}
        has_noise = any(label_sizes[i] < self.MIN_CLUSTER_SIZE for i in label_sizes)

        if not has_noise:
            return initial_labels

        # If we have noise, we need to re-cluster.
        # 1. First mark vectors that should be reclustered and vectors that should be ignored
        should_recluster = [
            label_sizes[label] >= self.MIN_CLUSTER_SIZE for label in initial_labels
        ]
        vectors_to_recluster = [vector for vector, flag in zip(vectors, should_recluster) if flag]

        # 2. Re-cluster
        self.k = None
        if vectors_to_recluster:
            new_labels = self.cluster(vectors_to_recluster)
        else:
            new_labels = []

        # 3. Reconstruct labels in original order and size
        i = 0
        full_labels = []
        for flag in should_recluster:
            if flag and i < len(new_labels):
                full_labels.append(new_labels[i])
                i += 1
            else:
                full_labels.append(-1)
        return full_labels


class HdbscanClustering(BaseClustering):
    """
    HDBSCAN Clustering wrapper class. Performs clustering using hdbscan's algorithm.
    """

    def __init__(self, min_cluster_size: int = 5):
        """Instantiates a new instance of the Hdbscan Clustering wrapper class"""

        logger.debug("Initializing HdbscanClustering with min_cluster_size=`%s`",
                     min_cluster_size)
        self.model = HDBSCAN(min_cluster_size=min_cluster_size)

    def cluster(self, vectors: List[List[float]]) -> List[int]:
        """Given a list of vectors, performs hdbscan based clustering and returns the output labels"""

        return self.model.fit_predict(vectors)


class OpticsClustering(BaseClustering):
    """
    OPTICS Clustering wrapper class. Performs clustering using optics's algorithm as implemented
    on sklearn.
    """

    def __init__(self, min_samples: int = 5):
        """Instantiates a new instance of the Optics Clustering wrapper class"""

        logger.debug("Initializing OpticsClustering with min_samples=`%s`",
                     min_samples)
        self.model = OPTICS(min_samples=min_samples, metric='euclidean', n_jobs=-1)

    def cluster(self, vectors: List[List[float]]) -> List[int]:
        """Given a list of vectors, performs hdbscan based clustering and returns the output labels"""

        self.model.fit_predict(vectors)
        return self.model.labels_


class AffinityPropagationClustering(BaseClustering):
    """
    Affinity Propagation Clustering wrapper class. Performs clustering using optics's algorithm as implemented
    on sklearn.
    """

    def __init__(self):
        """Instantiates a new instance of the Affinity Propagation Clustering wrapper class"""

        logger.debug("Initializing AffinityPropagationClustering")
        self.model = AffinityPropagation(random_state=self.RANDOM_SEED)

    def cluster(self, vectors: List[List[float]]) -> List[int]:
        """Given a list of vectors, performs hdbscan based clustering and returns the output labels"""

        self.model.fit_predict(vectors)
        return self.model.labels_


class DBSCANClustering(BaseClustering):
    """
    DBSCAN Clustering wrapper class. Performs clustering using optics's algorithm as implemented
    on sklearn.
    """

    def __init__(self):
        """Instantiates a new instance of the DBSCAN Clustering wrapper class"""

        logger.debug("Initializing DBSCANClustering")
        self.model = DBSCAN(eps=0.20, min_samples=5, n_jobs=-1)

    def cluster(self, vectors: List[List[float]]) -> List[int]:
        """Given a list of vectors, performs dbscan based clustering and returns the output labels"""

        self.model.fit_predict(vectors)
        return self.model.labels_


class MeanShiftClustering(BaseClustering):
    """
    Mean Shift Clustering wrapper class. Performs clustering using optics's algorithm as implemented
    on sklearn.
    """

    def __init__(self):
        """Instantiates a new instance of the Mean Shift Clustering wrapper class"""

        logger.debug("Initializing MeanShiftClustering")
        self.model = MeanShift(n_jobs=-1)

    def cluster(self, vectors: List[List[float]]) -> List[int]:
        """Given a list of vectors, performs hdbscan based clustering and returns the output labels"""

        self.model.fit_predict(vectors)
        return self.model.labels_


class SpectClustering(BaseClustering):
    """
    Mean Shift Clustering wrapper class. Performs clustering using optics's algorithm as implemented
    on sklearn.  If not specified, automatically detects the proper value of
    K that fits the algorithm best.
    """

    MIN_K = 2
    MAX_K = 4

    def __init__(self, k: Optional[int] = None) -> None:
        """Initializes a new instance of the KMeans clustering algorithm"""

        if k is not None:
            if not self.MIN_K <= k <= self.MAX_K:
                raise ValueError(k)

            self.k = k
            self.model = self.instantiate_model(self.k)
            logger.debug("Initializing SpectClustering with k=`%s`", self.k)
        else:
            self.k = None
            logger.debug("Deferring Spectral Clustering model initialization until vectors are known")

    def instantiate_model(self, k: int) -> SpectralClustering:
        """Instantiates Sklearn's KMeans class and stores within wrapper class"""

        return SpectralClustering(n_clusters=k, n_jobs=-1, random_state=self.RANDOM_SEED)

    def cluster(self, vectors: List[List[float]]) -> List[int]:
        """Given a list of vectors, performs kmeans based clustering and returns labels of the output"""

        # Easy case: we know `k` beforehand
        if self.k is not None:
            return self.model.fit_predict(vectors)

        # Determining best K
        sil_scores = {}
        results = {}
        for k in range(self.MIN_K, self.MAX_K + 1):
            model = self.instantiate_model(k)
            output = model.fit_predict(vectors)
            sil_score = silhouette_score(vectors, output)
            sil_scores[k] = sil_score
            results[k] = output

        # Settings best K
        max_sil = sorted(sil_scores.items(), key=lambda x: x[1], reverse=True)[0]
        self.k = max_sil[0]
        self.model = self.instantiate_model(self.k)
        logger.debug("Maximum silhouette score achieved with k=%s (silhouette score: %s)", max_sil[0], max_sil[1])

        # Recursive call will get the real result
        return results[k]


def get_clustering_functions() -> Dict[str, BaseClustering]:
    """
    Returns a dictionary of the available clustering algorithms,
    indexable by their names.
    """

    return {
        'K-means': KMeansClustering(),
        'Mean-Shift': MeanShiftClustering(),
        'Affinity': AffinityPropagationClustering(),
        'DBSCAN': DBSCANClustering(),
        'HDBSCAN': HdbscanClustering(),
        'OPTICS': OpticsClustering(),
        # 'Spectral': SpectClustering(),
    }


def get_clusters_from_labels(texts: List[str], labels: List[int]) -> List[List[str]]:
    """Given a series of texts and their labels (as produced during clustering), reconstructs the clusters"""

    return [[text for text, lab in zip(texts, labels) if lab == label] for label in set(labels) if label != -1]


def main() -> None:
    """Runs a test script in order to check the behavior of clustering algorithms"""

    vectors = [np.random.random(3) for _ in range(150)]

    results = {}
    for name, algorithm in get_clustering_functions().items():
        output = algorithm.cluster(vectors)
        logger.debug(f"{name} result: {output}")
        try:
            sil_score = silhouette_score(vectors, output)
            logger.debug(f"{name} sil score: {sil_score}")
            results[name] = sil_score
        except ValueError:
            logger.error(f"couldn't get sil score for {name}")

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    logger.debug("silhouette score rsults")
    for name, sil_score in sorted_results:
        logger.debug(f"{name}: {sil_score}")


if __name__ == "__main__":
    main()
