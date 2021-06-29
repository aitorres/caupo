"""
Main script that performs clustering experiments taking tweets stored on a
database as the main corpus.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from caupo.ngrams import get_top_ngrams
from caupo.clustering import get_clustering_functions, get_clusters_from_labels
from caupo.database import (get_results_collection, result_already_exists,
                            transform_types_for_database)
from caupo.embeddings import get_embedder_functions
from caupo.preprocessing import preprocess_v2
from caupo.sentiment import calculate_average_sentiment
from caupo.tags import Tag, fetch_tag_from_db, get_tags_by_frequency
from caupo.utils import get_main_corpus, plot_clusters

logger = logging.getLogger("caupo")

BASE_OUTPUT_FOLDER = "outputs/cluster_tags/"


def create_output_files(frequency: str) -> None:
    """Given a frequency, creates files with headers (if needed) to store output info"""

    output_folder = f"{BASE_OUTPUT_FOLDER}/{frequency}"
    os.makedirs(output_folder, exist_ok=True)

    csv_file = Path(f"{output_folder}/results.csv")
    if not csv_file.exists():
        with open(csv_file, "w") as file_handler:
            file_handler.write("frequency,tag,embedder,algorithm,time,n_clusters,has_outliers," +
                               "tweets,valid_tweets,outliers,noise_percentage,avg_cluster_size," +
                               "min_cluster_size,max_cluster_size,sil_score,db_score,ch_score\n")

    md_file = Path(f"{output_folder}/results.md")
    if not md_file.exists():
        with open(md_file, "w") as file_handler:
            file_handler.write(
                "|Frequency|Tag|Embedder|Algorithm|Time (s)|Amount of Clusters|Has Outliers|" +
                "Tweets|Valid Tweets|Outliers|Noise Percentage|Avg. Cluster Size |Min. Cluster Size|" +
                "Max. Cluster Size|Silhouette Score|Davies-Bouldin Score|Calinski-Harabasz Score|\n" +
                "|---|---|---|----|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    return csv_file, md_file


def cluster_tag(tag: Tag, embedder_functions: Dict[str, Callable[[List[str]], List[float]]],
                frequency: str, csv_file: Path, md_file: Path) -> None:
    """
    Given an entity tag, performs clustering and reports result to logs
    """

    results_collection = get_results_collection()

    # Extracting tweets
    logger.debug("Extracting tweets from tags")
    tweets = tag["tweets"]
    tag_name = tag["tag"]

    # Normalizing tweets
    logger.debug("Cleaning tweets")
    cleaned_tweets = list(set(map(preprocess_v2, tweets)))
    cleaned_tweets = [c for c in cleaned_tweets if len(c) > 0]
    logger.info("Collection of cleaned tweets has a size of %s tweets", len(cleaned_tweets))

    # Applying clustering and reporting tweets
    logger.debug("All ready, starting experiments!")
    sil_scores = {}
    db_scores = {}
    ch_scores = {}
    for embedder_name, embedder in embedder_functions.items():
        print()
        logger.info("[%s] Now trying embedder %s", tag_name, embedder_name)
        vectors = embedder(cleaned_tweets)
        algorithms = get_clustering_functions()
        for algorithm_name, algorithm in algorithms.items():

            if result_already_exists(frequency, tag_name, algorithm_name, embedder_name):
                logger.debug("Skipping combination %s, %s, %s and %s since it already exists", frequency, tag_name,
                             algorithm_name, embedder_name)
                continue

            t1 = -1
            logger.info("[%s] Now trying algorithm %s with embedder %s", tag_name, algorithm_name, embedder_name)
            output_folder = f"{BASE_OUTPUT_FOLDER}/{frequency}/{embedder_name}/{algorithm_name}"
            os.makedirs(output_folder, exist_ok=True)

            try:
                t0 = time.time()
                labels = algorithm.cluster(vectors)
                t1 = time.time() - t0
                if isinstance(labels, np.ndarray):
                    labels = labels.tolist()
                logger.info("[%s] Clustering produced %s distinct labels: %s", tag_name, len(set(labels)), set(labels))
                labels = algorithm.remove_small_clusters(labels)
                logger.info("[%s] After removing small clusters, clustering produced %s distinct labels: %s",
                            tag_name, len(set(labels)), set(labels))
            except ValueError:
                labels = []
                logger.warning("[%s] Couldn't produce clusterings with algorithm %s", tag_name, algorithm_name)

            # If clusterization happened properly, produce outputs and compute scores
            if -1 in labels:
                logger.info("[%s] This clusterization found %s outliers (out of %s elements)",
                            tag_name, len([label for label in labels if label == -1]), len(labels))

            if len(set([label for label in labels if label != -1])) == 0:
                logger.info("[%s] Skipping plots and computations since no real clusters were found", tag_name)
                results_collection.insert_one(transform_types_for_database({
                    'frequency': frequency,
                    'tag': tag_name,
                    'algorithm': algorithm_name,
                    'embedder': embedder_name,
                    'success': False,
                    'time': t1,
                    'labels': labels,
                    'amountCleanLabels': None,
                    'hasNoise': None,
                    'tweetsAmount': len(vectors),
                    'validTweetsAmount': None,
                    'noiseAmount': None,
                    'noisePercentage': None,
                    'clusters': None,
                    'avgClusterSize': None,
                    'minClusterSize': None,
                    'maxClusterSize': None,
                    'clusterThemes': None,
                    'averageSentiment': None,
                    'scores': {
                        'silhouette': None,
                        'davies_bouldin': None,
                        'calinski-harabasz': None,
                    },
                }))
                # Storing output in CSV File
                with open(csv_file, "a") as file_handler:
                    file_handler.write(
                        f"{frequency},{tag['tag']},{embedder_name},{algorithm_name},{t1},{None}," +
                        f"{None},{len(vectors)},{None},{None}," +
                        f"{None},{None},{None},{None}," +
                        f"{None},{None},{None}\n")

                # Storing output to Markdown file
                with open(md_file, "a") as file_handler:
                    file_handler.write(
                        f"|{frequency}|{tag['tag']}|{embedder_name}|{algorithm_name}|{t1}|{None}|" +
                        f"{None}|{len(vectors)}|{None}|{None}|" +
                        f"{None}|{None}|{None}|{None}|" +
                        f"{None}|{None}|{None}|\n")
                continue

            # Cleaning elements from outliers
            clean_elements = [(vector, label) for vector, label in zip(vectors, labels) if label != -1]
            clean_vectors = [elem[0] for elem in clean_elements]
            clean_labels = [elem[1] for elem in clean_elements]

            # Plotting clusters
            logger.info("[%s] Plotting %s clusters", tag_name, len(set(labels)))
            plot_clusters(vectors, f"{output_folder}/{tag['tag']}_plot.png",
                          f"{algorithm_name} - {embedder_name} (n={len(set(labels))})",
                          labels=labels, plot_outliers=True)

            if -1 in labels:
                logger.info("[%s] Plotting %s clean clusters (no outliers)", tag_name, len(set(clean_labels)))
                plot_clusters(vectors, f"{output_folder}/{tag['tag']}_plot_clean.png",
                              f"{algorithm_name} - {embedder_name} (no outliers, n={len(set(labels))})",
                              labels=labels, plot_outliers=False)

            # If we got more than one cluster, compute results
            if len(set(clean_labels)) > 1:
                logger.info("[%s] This clusterization produced %s successfully clustered elements into %s clusters",
                            tag_name, len(clean_elements), len(set(clean_labels)))
                for label in set(clean_labels):
                    cluster_length = len([cln_lab for cln_lab in clean_labels if cln_lab == label])
                    logger.debug("[%s] Cluster %s: %s elements", tag_name, label, cluster_length)
                    if cluster_length == 1:
                        cluster = [tweet for tweet, lab in zip(cleaned_tweets, labels) if lab == label]
                        logger.debug("[%s] Cluster of length %s made of this tweet: %s",
                                     tag_name, cluster_length, cluster[0])

                sil_score = silhouette_score(clean_vectors, clean_labels)
                logger.info("[%s] This clusterization got a silhouette score of %s", tag_name, sil_score)
                sil_scores[(embedder_name, algorithm_name, len(set(clean_labels)))] = sil_score

                db_score = davies_bouldin_score(clean_vectors, clean_labels)
                logger.info("[%s] This clusterization got a Davies-Bouldin index of %s", tag_name, db_score)
                db_scores[(embedder_name, algorithm_name, len(set(clean_labels)))] = db_score

                ch_score = calinski_harabasz_score(clean_vectors, clean_labels)
                logger.info("[%s] This clusterization got a Calisnki-Harabasz score of %s", tag_name, ch_score)
                ch_scores[(embedder_name, algorithm_name, len(set(clean_labels)))] = ch_score
            else:
                logger.warning("[%s] Skipping calculation of scores for %s using %s",
                               tag_name, algorithm_name, embedder_name)
                sil_score = None
                db_score = None
                ch_score = None

            logger.info("[%s] Labelling clusters with most frequent bigrams and calculating sentiment",
                        tag_name)
            tweet_clusters = get_clusters_from_labels(cleaned_tweets, labels)

            top_n_amount_per_frequency = {
                'daily': 5,
                'weekly': 8,
                'monthly': 10,
            }
            top_n_amount = top_n_amount_per_frequency[frequency]
            cluster_themes = {}
            average_sentiment = {}
            for idx, tweet_cluster in enumerate(tweet_clusters):
                cluster_size = len(tweet_cluster)
                cluster_bigram = get_top_ngrams(tweet_cluster, top_n_amount=top_n_amount, ngram_size=2)
                logger.info("[%s] Cluster `%s` (size: %s) has a theme of: %s",
                            tag_name, idx, cluster_size, cluster_bigram)
                cluster_themes[str(idx)] = cluster_bigram

                cluster_sentiment = calculate_average_sentiment(tweet_cluster)
                logger.info("[%s] Cluster `%s` (size: %s) has an average sentiment of: %s",
                            tag_name, idx, cluster_size, cluster_sentiment)
                average_sentiment[str(idx)] = cluster_sentiment

            clusters = {
                str(label): [tweet for tweet_label, tweet in zip(labels, tweets) if tweet_label == label]
                for label in set(clean_labels)
            }
            noise_percentage = round((len(vectors) - len(clean_vectors)) / len(vectors), 2)
            cluster_sizes = [len(cluster) for cluster in clusters.values()]
            avg_cluster_size = round(sum(cluster_sizes) / len(clusters.keys()), 2)
            min_cluster_size = min(cluster_sizes)
            max_cluster_size = max(cluster_sizes)

            # Storing output in CSV File
            with open(csv_file, "a") as file_handler:
                file_handler.write(
                    f"{frequency},{tag['tag']},{embedder_name},{algorithm_name},{t1},{len(set(clean_labels))}," +
                    f"{-1 in labels},{len(vectors)},{len(clean_vectors)},{len(vectors) - len(clean_vectors)}," +
                    f"{noise_percentage},{avg_cluster_size},{min_cluster_size},{max_cluster_size}," +
                    f"{sil_score},{db_score},{ch_score}\n")

            # Storing output to Markdown file
            with open(md_file, "a") as file_handler:
                file_handler.write(
                    f"|{frequency}|{tag['tag']}|{embedder_name}|{algorithm_name}|{t1}|{len(set(clean_labels))}|" +
                    f"{-1 in labels}|{len(vectors)}|{len(clean_vectors)}|{len(vectors) - len(clean_vectors)}|" +
                    f"{noise_percentage}|{avg_cluster_size}|{min_cluster_size}|{max_cluster_size}|" +
                    f"{sil_score}|{db_score}|{ch_score}|\n")

            # Storing results of this run to database
            logger.debug("[%s] Storing results to database...", tag_name)
            results_collection.insert_one(transform_types_for_database({
                'frequency': frequency,
                'tag': tag_name,
                'algorithm': algorithm_name,
                'embedder': embedder_name,
                'success': True,
                'time': t1,
                'labels': labels,
                'amountCleanLabels': len(set(clean_labels)),
                'hasNoise': -1 in labels,
                'tweetsAmount': len(vectors),
                'validTweetsAmount': len(clean_vectors),
                'noiseAmount': len(vectors) - len(clean_vectors),
                'noisePercentage': noise_percentage,
                'avgClusterSize': avg_cluster_size,
                'minClusterSize': min_cluster_size,
                'maxClusterSize': max_cluster_size,
                'clusterThemes': cluster_themes,
                'averageSentiment': average_sentiment,
                'scores': {
                    'silhouette': sil_score,
                    'davies_bouldin': db_score,
                    'calinski_harabasz': ch_score,
                },
            }))
            logger.debug("[%s] Results stored!", tag_name)

    # https://en.wikipedia.org/wiki/Silhouette_(clustering)
    sorted_sil_scores = sorted(sil_scores.items(), key=lambda x: x[1], reverse=True)
    logger.debug("Silhouette score results (sort: desc, higher is better)")
    for score_tag, score in sorted_sil_scores:
        logger.debug(f"{score_tag}: {score}")
    print()

    # https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
    sorted_db_scores = sorted(db_scores.items(), key=lambda x: x[1])
    logger.debug("Davies-Boulding score results (sort: asc, less is better)")
    for score_tag, score in sorted_db_scores:
        logger.debug(f"{score_tag}: {score}")
    print()

    sorted_ch_scores = sorted(ch_scores.items(), key=lambda x: x[1], reverse=True)
    logger.debug("Calinski-Harabasz score results (sort: desc, higher is better)")
    for score_tag, score in sorted_ch_scores:
        logger.debug(f"{score_tag}: {score}")
    print()


def main() -> None:
    """
    Main script that extracts input arguments and runs the script
    """

    parser = argparse.ArgumentParser(description="Performs clustering over tweets stored on the database")
    parser.add_argument("frequency", metavar="FREQ", type=str, choices=["daily", "weekly", "monthly"])
    args = parser.parse_args()

    # Get main corpus for recreating embedders
    logger.debug("Getting main corpus for training embedders")
    corpus = get_main_corpus()

    logger.debug("Cleaning corpus")
    cleaned_corpus = list(set(map(preprocess_v2, corpus)))
    cleaned_corpus = [c for c in cleaned_corpus if len(c) > 0]
    logger.info("Cleaned corpus has %s tweets", len(cleaned_corpus))

    # Getting vector embedders
    logger.debug("Initializing embedders")
    embedder_functions = get_embedder_functions(cleaned_corpus)

    logger.debug("Getting all tags with `%s` frequency", args.frequency)
    tags = get_tags_by_frequency(args.frequency)
    csv_file, md_file = create_output_files(args.frequency)

    # We traverse the iterable in reverse order
    for tag_name, _ in tags[::-1]:
        logger.debug("Fetching tag `%s` from database", tag_name)
        tag = fetch_tag_from_db(args.frequency, tag_name)
        if tag is None:
            logger.warning("Tag %s has not been found in the database, skipping...")
            continue
        cluster_tag(tag, embedder_functions, args.frequency, csv_file, md_file)
        logger.debug("Finished work in tag `%s`", tag_name)


if __name__ == "__main__":
    main()
