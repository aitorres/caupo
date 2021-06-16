"""
Main script that performs clustering experiments taking tweets stored on a
database as the main corpus.
"""

import argparse
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from emoji import get_emoji_regexp
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from caupo.clustering import get_clustering_functions, get_clusters_from_labels
from caupo.embeddings import get_embedder_functions
from caupo.database import get_results_collection, result_already_exists, transform_types_for_database
from caupo.preprocessing import get_stopwords, map_strange_characters
from caupo.tags import Tag, fetch_tag_from_db, get_tags_by_frequency
from caupo.topic_modelling import get_topic_models, get_topics_from_model
from caupo.utils import get_main_corpus, plot_clusters, plot_top_words

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
            lambda x: not x.startswith("@") and not x.isdigit() and not x[0].isdigit() and not x[-1].isdigit(),
            tweet.split()
        )
    )
    tweet = " ".join(re.sub(get_emoji_regexp(), "", tweet).split())
    base_tweet = " ".join(re.sub(r'[0-9#@:;_\-+=/°¿?¡%!\"\'.,\[\]\\\(\)&]', ' ', tweet).split())

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
    cleaned_tweets = list(set(map(quick_preprocess, tweets)))
    logger.info("Collection of cleaned tweets has a size of %s tweets", len(cleaned_tweets))

    # Applying clustering and reporting tweets
    logger.debug("All ready, starting experiments!")
    sil_scores = {}
    db_scores = {}
    ch_scores = {}
    for embedder_name, embedder in embedder_functions.items():
        logger.info("Now trying embedder %s", embedder_name)
        vectors = embedder(cleaned_tweets)
        algorithms = get_clustering_functions()
        for algorithm_name, algorithm in algorithms.items():

            if result_already_exists(frequency, tag_name, algorithm_name, embedder_name):
                logger.debug("Skipping combination %s, %s, %s and %s since it already exists", frequency, tag_name,
                             algorithm_name, embedder_name)
                continue

            t1 = -1
            topics_list = None
            logger.info("Now trying algorithm %s with embedder %s", algorithm_name, embedder_name)
            output_folder = f"{BASE_OUTPUT_FOLDER}/{frequency}/{embedder_name}/{algorithm_name}"
            os.makedirs(output_folder, exist_ok=True)

            try:
                t0 = time.time()
                labels = algorithm.cluster(vectors)
                t1 = time.time() - t0
                if isinstance(labels, np.ndarray):
                    labels = labels.tolist()
                logger.info("Clustering produced %s distinct labels: %s", len(set(labels)), set(labels))
                labels = algorithm.remove_small_clusters(labels)
                logger.info("After removing small clusters, clustering produced %s distinct labels: %s",
                            len(set(labels)), set(labels))
            except ValueError:
                labels = []
                logger.warning("Couldn't produce clusterings with algorithm %s", algorithm_name)

            # If clusterization happened properly, produce outputs and compute scores
            if -1 in labels:
                logger.info("This clusterization found %s outliers (out of %s elements)",
                            len([label for label in labels if label == -1]), len(labels))

            if len(set([label for label in labels if label != -1])) == 0:
                logger.info("Skipping plots and computations since no real clusters were found")
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
                    'avgClusterSize': -1,
                    'minClusterSize': -1,
                    'maxClusterSize': None,
                    'scores': {
                        'silhouette': None,
                        'davies_bouldin': None,
                    },
                    'topics': topics_list,
                }))
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

                ch_score = calinski_harabasz_score(clean_vectors, clean_labels)
                logger.info("This clusterization got a Calisnki-Harabasz score of %s", ch_score)
                ch_scores[(embedder_name, algorithm_name, len(set(clean_labels)))] = ch_score
            else:
                logger.warning("Skipping calculation of scores for %s using %s", algorithm_name, embedder_name)
                sil_score = None
                db_score = None
                ch_score = None

            logger.info("Starting topics generation")
            tweet_clusters = get_clusters_from_labels(cleaned_tweets, labels)
            topics_list = []
            for topic_model_name, topic_model in get_topic_models().items():
                logger.info("Now trying %s", topic_model_name)
                topic_dict = {
                    'model': topic_model_name,
                    'topics_per_cluster': [],
                }
                for idx, tweet_cluster in enumerate(tweet_clusters):
                    if len(tweet_cluster) < algorithm.MIN_CLUSTER_SIZE:
                        logger.warning("Ignoring cluster %s since it's only got %s elements", idx, len(tweet_cluster))
                        topic_dict['topics_per_cluster'].append({
                            'cluster_id': idx,
                            'topics': None,
                        })
                        continue
                    min_word_length_for_topics = 3
                    topics_amount = 1
                    top_words_amount = 6
                    try:
                        tweet_cluster_for_topics = list(
                            map(
                                lambda t: " ".join([w for w in t.split() if len(w) >= min_word_length_for_topics]),
                                tweet_cluster
                            )
                        )
                        model, feature_names = topic_model(tweet_cluster_for_topics, topics_amount)
                        topics = get_topics_from_model(model, top_words_amount, feature_names)
                        logger.info("Topics for cluster %s (length is %s): %s", idx, len(tweet_cluster), topics)
                        topic_dict['topics_per_cluster'].append({
                            'cluster_id': idx,
                            'topics': [
                                [
                                    {
                                        'keyword': keyword,
                                        'weight': weight,
                                    } for keyword, weight in topic
                                ] for topic in topics]
                        })
                        topics_list.append(topic_dict)
                        plot_top_words(model, feature_names, top_words_amount,
                                       f"Cluster {idx} - {algorithm_name} - {embedder_name}",
                                       f"{output_folder}/{tag['tag']}_cluster_{idx}_topics_{topic_model_name}.png")
                    except ValueError:
                        logger.warning("Error during topic modelling on cluster %s", idx)

            clusters = [
                {str(label): [tweet for tweet_label, tweet in zip(labels, tweets) if tweet_label == label]}
                for label in set(clean_labels)
            ]
            noise_percentage = round(len(vectors) - len(clean_vectors) / len(vectors), 2)
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
            logger.debug("Storing results to database...")
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
                'clusters': clusters,
                'avgClusterSize': avg_cluster_size,
                'minClusterSize': min_cluster_size,
                'maxClusterSize': max_cluster_size,
                'scores': {
                    'silhouette': sil_score,
                    'davies_bouldin': db_score,
                    'calinski_harabasz': ch_score,
                },
                'topics': topics_list,
            }))
            logger.debug("Results stored!")

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
    cleaned_corpus = list(set(map(quick_preprocess, corpus)))
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
        cluster_tag(tag, embedder_functions, args.frequency, csv_file, md_file)
        logger.debug("Finished work in tag `%s`", tag_name)


if __name__ == "__main__":
    main()
