"""
Main script that performs clustering experiments taking tweets stored on a
database as the main corpus.
"""

import argparse
import logging

from caupo.clustering import get_clustering_functions
from caupo.embeddings import get_embedder_functions
from caupo.tags import Tag, get_tags_by_frequency, fetch_tag_from_db
from caupo.preprocessing import map_strange_characters, get_stopwords
from caupo.utils import get_main_corpus
from sklearn.metrics import silhouette_score, davies_bouldin_score

logger = logging.getLogger("caupo")


def quick_preprocess(tweet: str) -> str:
    """
    Quick prototype of preprocessing
    # TODO: abstract this function and normalize with what is done on entity extractor
    """

    stopwords = get_stopwords()

    cleaned_tweet = " ".join(
        list(
            map(
                lambda t: "" if t in stopwords else t,
                map_strange_characters(
                    tweet.lower()
                ).split()
            )
        )
    )
    return cleaned_tweet


def cluster_tag(tag: Tag) -> None:
    """
    Given an entity tag, performs clustering and reports result to logs
    # TODO: Store on tags
    # TODO: Refactor for efficiency and resource management
    """

    # Get main corpus for recreating embedders
    logger.debug("Getting main corpus for training embedders")
    corpus = get_main_corpus()

    # Extracting tweets
    logger.debug("Extracting tweets from tags")
    tweets = tag["tweets"]

    # Normalizing tweets
    logger.debug("Cleaning corpus")
    cleaned_corpus = list(map(quick_preprocess, corpus))
    logger.info("Cleaned corpus has %s tweets", len(cleaned_corpus))

    logger.debug("Cleaning tweets")
    cleaned_tweets = list(map(quick_preprocess, tweets))
    logger.info("Collection of cleaned tweets has a size of %s tweets", len(cleaned_tweets))

    # Getting vector representations
    logger.debug("Initializing embedders")
    embedder_functions = get_embedder_functions(cleaned_corpus)

    # Applying clustering and reporting tweets
    logger.debug("All ready, starting experiments!")
    sil_scores = {}
    db_scores = {}
    for embedder_name, embedder in embedder_functions.items():
        logger.info("Now trying embedder %s", embedder_name)
        vectors = embedder(cleaned_tweets)
        algorithms = get_clustering_functions()
        for algorithm_name, algorithm in algorithms.items():
            logger.info("Now trying algorithm %s with embedder %s", algorithm_name, embedder_name)
            try:
                labels = algorithm.cluster(vectors)
                logger.info("Clustering produced %s distinct labels: %s", len(set(labels)), set(labels))
            except ValueError:
                labels = []
                logger.warning("Couldn't produce clusterings with algorithm %s", algorithm_name)

            if -1 in labels:
                logger.info("This clusterization found %s outliers (out of %s elements)",
                            len([label for label in labels if label == -1]), len(labels))

            clean_elements = [
                (vector, label) for vector, label in zip(vectors, labels) if label != -1
            ]
            clean_vectors = [elem[0] for elem in clean_elements]
            clean_labels = [elem[1] for elem in clean_elements]

            if len(set(labels)) > 1:
                logger.info("This clusterization produced %s successfully clustered elements", len(clean_elements))

                sil_score = silhouette_score(clean_vectors, clean_labels)
                logger.info("This clusterization got a silhouette score of %s", sil_score)
                sil_scores[(embedder_name, algorithm_name)] = sil_score

                db_score = davies_bouldin_score(clean_vectors, clean_labels)
                logger.info("This clusterization got a Davies-Bouldin index of %s", db_score)
                db_scores[(embedder_name, algorithm_name)] = db_score
            else:
                logger.warning("Skipping calculation of scores for %s using %s", algorithm_name, embedder_name)

    # https://en.wikipedia.org/wiki/Silhouette_(clustering)
    sorted_sil_scores = sorted(sil_scores.items(), key=lambda x: x[1], reverse=True)
    logger.debug("Silhouette score results (sort: desc, higher is better)")
    for score_tag, score in sorted_sil_scores:
        logger.debug(f"{score_tag}: {score}")

    # https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
    sorted_db_scores = sorted(db_scores.items(), key=lambda x: x[1])
    logger.debug("Davies-Boulding score results (sort: asc, less is better)")
    for score_tag, score in sorted_db_scores:
        logger.debug(f"{score_tag}: {score}")


def main() -> None:
    """
    Main script that extracts input arguments and runs the script
    """

    parser = argparse.ArgumentParser(description="Performs clustering over tweets stored on the database")
    parser.add_argument("frequency", metavar="FREQ", type=str, choices=["daily", "weekly", "monthly"])
    args = parser.parse_args()

    logger.debug("Getting all tags with `%s` frequency", args.frequency)
    tags = get_tags_by_frequency(args.frequency)

    #! TODO: rework script
    for tag_name, _ in tags[:1]:
        logger.debug("Fetching tag `%s` from database", tag_name)
        tag = fetch_tag_from_db(args.frequency, tag_name)
        cluster_tag(tag)


if __name__ == "__main__":
    main()
