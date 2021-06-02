"""
Main script that performs clustering experiments taking tweets stored on a
database as the main corpus.
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from emoji import UNICODE_EMOJI
from sklearn.metrics import davies_bouldin_score, silhouette_score

from caupo.clustering import get_clustering_functions
from caupo.embeddings import get_embedder_functions
from caupo.preprocessing import get_stopwords, map_strange_characters
from caupo.tags import (Tag, exclude_preexisting_tags, fetch_tag_from_db,
                        get_collection_by_frequency, get_tags_by_frequency)
from caupo.utils import get_main_corpus, plot_clusters

logger = logging.getLogger("caupo")

BASE_OUTPUT_FOLDER = "outputs/cluster_tags/"


def quick_preprocess(tweet: str) -> str:
    """
    Quick prototype of preprocessing
    # TODO: abstract this function and normalize with what is done on entity extractor
    """

    stopwords = get_stopwords()

    tweet = " ".join(
        filter(
            lambda x: not x.startswith("@") and not set(x) == {"j", "a"} and not x.isdigit() and x not in UNICODE_EMOJI['en'],
            tweet.split()
        )
    )

    base_tweet = re.sub(r'[#@:;_\-+=/°¿?¡%!\"\'.,\[\]\\\(\)&]', ' ', tweet)

    cleaned_tweet = " ".join(
        list(
            map(
                lambda t: "" if t in stopwords else t,
                map_strange_characters(
                    base_tweet.lower()
                ).split()
            )
        )
    )
    return " ".join(cleaned_tweet.split())


def create_output_files(frequency: str) -> None:
    """Given a frequency, creates files with headers (if needed) to store output info"""

    output_folder = f"{BASE_OUTPUT_FOLDER}/{frequency}"
    os.makedirs(output_folder, exist_ok=True)

    csv_file = Path(f"{output_folder}/results.csv")
    if not csv_file.exists():
        with open(csv_file, "w") as file_handler:
            file_handler.write("frequency,tag,embedder,algorithm,n_clusters,has_outliers," +
                               "tweets,valid_tweets,outliers,sil_score,db_score\n")

    md_file = Path(f"{output_folder}/results.md")
    if not md_file.exists():
        with open(md_file, "w") as file_handler:
            file_handler.write(
                "|Frequency|Tag|Embedder|Algorithm|Amount of Clusters|Has Outliers|" +
                "Tweets|Valid Tweets|Outliers|Silhouette Score|Davies-Bouldin Score|\n" +
                "|---|---|---|---|---|---|---|---|---|---|---|\n")

    return csv_file, md_file


def cluster_tag(tag: Tag, frequency: str, csv_file: Path, md_file: Path) -> None:
    """
    Given an entity tag, performs clustering and reports result to logs
    # TODO: Store on tags
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
    clusters_info = []
    for embedder_name, embedder in embedder_functions.items():
        logger.info("Now trying embedder %s", embedder_name)
        vectors = embedder(cleaned_tweets)
        algorithms = get_clustering_functions()
        for algorithm_name, algorithm in algorithms.items():
            logger.info("Now trying algorithm %s with embedder %s", algorithm_name, embedder_name)
            output_folder = f"{BASE_OUTPUT_FOLDER}/{frequency}/{embedder_name}/{algorithm_name}"
            os.makedirs(output_folder, exist_ok=True)

            try:
                labels = algorithm.cluster(vectors)
                if isinstance(labels, np.ndarray):
                    labels = labels.tolist()
                logger.info("Clustering produced %s distinct labels: %s", len(set(labels)), set(labels))
            except ValueError:
                labels = []
                logger.warning("Couldn't produce clusterings with algorithm %s", algorithm_name)

            # If clusterization happened properly, produce outputs and compute scores
            if -1 in labels:
                logger.info("This clusterization found %s outliers (out of %s elements)",
                            len([label for label in labels if label == -1]), len(labels))

            if len(set([label for label in labels if label != -1])) == 0:
                logger.info("Skipping plots and computations since no real clusters were found")
                clusters_info.append({
                    'algorithm': algorithm_name,
                    'embedder': embedder_name,
                    'success': False,
                    'labels': None,
                    'clusters': None,
                    'scores': None,
                    'topics': None,
                })
                continue

            # Cleaning elements from outliers
            clean_elements = [(vector, label) for vector, label in zip(vectors, labels) if label != -1]
            clean_vectors = [elem[0] for elem in clean_elements]
            clean_labels = [elem[1] for elem in clean_elements]

            # Plotting clusters
            logger.info("Plotting clusters")
            plot_clusters(vectors, f"{output_folder}/{tag['tag']}_plot.png",
                          f"{algorithm_name} - {embedder_name} (n={len(set(labels))})",
                          labels=labels, plot_outliers=True)

            if -1 in labels:
                logger.info("Plotting clean clusters (no outliers)")
                plot_clusters(vectors, f"{output_folder}/{tag['tag']}_plot_clean.png",
                              f"{algorithm_name} - {embedder_name} (no outliers, n={len(set(labels))})",
                              labels=labels, plot_outliers=False)

            # If we got more than one cluster, compute results
            if len(set(clean_labels)) > 1:
                logger.info("This clusterization produced %s successfully clustered elements into %s clusters",
                            len(clean_elements), len(set(clean_labels)))
                for label in set(clean_labels):
                    cluster_length = len([cln_lab for cln_lab in clean_labels if cln_lab == label])
                    logger.debug("Cluster %s: %s elements", label, cluster_length)
                    if cluster_length == 1:
                        cluster = [tweet for tweet, lab in zip(cleaned_tweets, labels) if lab == label]
                        logger.debug("Cluster of length %s made of this tweet: %s", cluster_length, cluster[0])

                sil_score = silhouette_score(clean_vectors, clean_labels)
                logger.info("This clusterization got a silhouette score of %s", sil_score)
                sil_scores[(embedder_name, algorithm_name, len(set(clean_labels)))] = sil_score

                db_score = davies_bouldin_score(clean_vectors, clean_labels)
                logger.info("This clusterization got a Davies-Bouldin index of %s", db_score)
                db_scores[(embedder_name, algorithm_name, len(set(clean_labels)))] = db_score
            else:
                logger.warning("Skipping calculation of scores for %s using %s", algorithm_name, embedder_name)
                sil_score = None
                db_score = None

            # Storing output in CSV File
            with open(csv_file, "a") as file_handler:
                file_handler.write(
                    f"{frequency},{tag['tag']},{embedder_name},{algorithm_name},{len(set(clean_labels))}," +
                    f"{-1 in labels},{len(vectors)},{len(clean_vectors)},{len(vectors) - len(clean_vectors)}," +
                    f"{sil_score},{db_score}\n")

            # Storing output to Markdown file
            with open(md_file, "a") as file_handler:
                file_handler.write(
                    f"|{frequency}|{tag['tag']}|{embedder_name}|{algorithm_name}|{len(set(clean_labels))}|" +
                    f"{-1 in labels}|{len(vectors)}|{len(clean_vectors)}|{len(vectors) - len(clean_vectors)}|" +
                    f"{sil_score}|{db_score}|\n")

            # Storing results of this run to database
            clusters = [
                {str(label): [tweet for tweet_label, tweet in zip(labels, tweets) if tweet_label == label]}
                for label in set(clean_labels)
            ]
            clusters_info.append({
                'algorithm': algorithm_name,
                'embedder': embedder_name,
                'success': True,
                'labels': labels,
                'clusters': clusters,
                'scores': {
                    'silhouette': sil_score,
                    'davies_bouldin': db_score,
                },
                'topics': None,  # TODO: generate topics
            })

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

    # Storing results to database
    logger.debug("Storing results to database...")
    collection = get_collection_by_frequency(frequency, prefix="clusters")
    db_object = {
        'frequency': frequency,
        'tag': tag,
        'tweets': tweets,
        'cleaned_tweets': cleaned_tweets,
        'tweets_amount': len(tweets),
        'clusters': clusters_info,
    }
    typed_object = transform_types(db_object)
    collection.insert_one(typed_object)
    logger.debug("Results stored!")


def transform_types(obj: Any) -> Any:
    """Given an object, transforms its type to a native Python type for storage in database"""

    # Numpy types
    if isinstance(obj, (np.generic)):
        return obj.item()

    # Collection types
    if isinstance(obj, (list, set)):
        return [transform_types(item) for item in obj]

    if isinstance(obj, np.ndarray):
        return transform_types(obj.tolist())

    # Structures
    if isinstance(obj, dict):
        return {key: transform_types(value) for key, value in obj.items()}

    return obj


def main() -> None:
    """
    Main script that extracts input arguments and runs the script
    """

    parser = argparse.ArgumentParser(description="Performs clustering over tweets stored on the database")
    parser.add_argument("frequency", metavar="FREQ", type=str, choices=["daily", "weekly", "monthly"])
    args = parser.parse_args()

    logger.debug("Getting all tags with `%s` frequency", args.frequency)
    tags = get_tags_by_frequency(args.frequency)
    tags = exclude_preexisting_tags(args.frequency, tags, prefix="clusters")
    csv_file, md_file = create_output_files(args.frequency)

    # ! TODO: rework script
    for tag_name, _ in tags[len(tags) - 1:]:
        logger.debug("Fetching tag `%s` from database", tag_name)
        tag = fetch_tag_from_db(args.frequency, tag_name)
        cluster_tag(tag, args.frequency, csv_file, md_file)


if __name__ == "__main__":
    main()
