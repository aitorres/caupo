"""
Module that runs a series of tests to measure clustering of tweets
with OPTICS, using different available word embeddings
"""

import time
import logging
import os
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from embeddings import get_embedder_functions
from preprocessing import preprocess_corpus
from utils import get_city_modes, get_text_from_all_tweets, plot_clusters, Timer

mpl.use('Agg')

# Instantiate logger
logger = logging.getLogger("caupo")
logger.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')

# Add console (standard output) handler to the logger
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Creating folder for output
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
BASE_OUTPUT_FOLDER = f"outputs/measure_optics_clusters/{ timestamp }"
os.makedirs(BASE_OUTPUT_FOLDER)

# Add file handler to the logger
file_handler = logging.FileHandler(f'{BASE_OUTPUT_FOLDER}/measure_optics_clusters.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


with Timer("Main script runtime"):
    city_modes = get_city_modes().items()
    embedders = get_embedder_functions().items()

    for city_mode_name, city_mode_tag in city_modes:
        with Timer(f"Starting evaluation for city mode `{city_mode_name}`"):
            with Timer("Getting tweets' text from database"):
                corpus = get_text_from_all_tweets(city=city_mode_tag)
            logger.info("Amount of tweets: %s", len(corpus))

            # Normalize tweets
            with Timer("Normalizing tweets' text"):
                preprocessed_corpus = preprocess_corpus(corpus)

            # Get rid of duplicate processed tweets (this should take care of duplicate, spammy tweets)
            with Timer("Removing duplicate tweets (bot protection)"):
                clean_corpus = list(set(preprocessed_corpus))
            logger.info("Amount of clean tweets: %s", len(clean_corpus))

        for embedder_name, embedder_function in embedders:
            with Timer(f"Getting vectors with embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                vectors = embedder_function(clean_corpus)

            with Timer(f"Getting reduced vectors (2D) for scatterplot with embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                pca_fit = PCA(n_components=2)
                scatterplot_vectors = pca_fit.fit_transform(vectors)

            OUTPUT_FOLDER = f"{BASE_OUTPUT_FOLDER}/{city_mode_name}/{embedder_name}"
            os.makedirs(OUTPUT_FOLDER)
            with open(f"{OUTPUT_FOLDER}/full_data.md", "a") as md_file:
                md_file.write("|City mode|Embedder|Distance metric|Time of Clustering|Silhouette score|Davies-Bouldin score|Calinski and Harabasz score|\n")
                md_file.write("|---------|--------|---------------|------------------|----------------|--------------------|---------------------------|\n")

            DISTANCE_METRICS = ["euclidean", "cosine"]
            MIN_SAMPLES = 10

            distance_time_dict = {}
            distance_silhouette_dict = {}
            distance_dav_boul_dict = {}
            distance_cal_har_dict = {}
            for distance_metric in DISTANCE_METRICS:
                logger.info("Starting evaluation of distance metric `%s`", distance_metric)

                with Timer(f"Finding clusters with distance metric `{distance_metric}` and embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                    optics = OPTICS(min_samples=MIN_SAMPLES, metric=distance_metric, n_jobs=-1)
                    t0 = time.time()
                    optics_result = optics.fit(vectors)
                    t1 = time.time()

                with Timer(f"Getting metrics with distance metric `{distance_metric}` and embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                    model_time = t1 - t0
                    optics_labels = optics_result.labels_
                    logger.info("OPTICS generated `%s` cluster(s)", len({i for i in optics_labels if i != -1}))

                    logger.debug("Calculating silhouette score")
                    sil_score = silhouette_score(vectors, optics_labels, metric=distance_metric)
                    logger.info("Silhouete score with distance metric `%s`: %s", distance_metric, sil_score)

                    logger.debug("Calculating Davies-Boulding score")
                    dav_boul_score = davies_bouldin_score(vectors, optics_labels)
                    logger.info("Davies-Boulding score with distance metric `%s`: %s", distance_metric, dav_boul_score)

                    logger.debug("Calculating Calinski & Harabasz score")
                    cal_har_score = calinski_harabasz_score(vectors, optics_labels)
                    logger.info("Calinski & Harabasz score with distance metric `%s`: %s", distance_metric, cal_har_score)

                with Timer(f"Storing results with distance metric `{distance_metric}` and embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                    # Time
                    with open(f"{OUTPUT_FOLDER}/time_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{distance_metric},{model_time}\n")
                    distance_time_dict[f"{distance_metric}"] = model_time

                    # Silhouette
                    with open(f"{OUTPUT_FOLDER}/silhouette_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{distance_metric},{sil_score}\n")
                    distance_silhouette_dict[f"{distance_metric}"] = sil_score

                    # Davies-Bouldin score
                    with open(f"{OUTPUT_FOLDER}/dav_boul_score_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{distance_metric},{dav_boul_score}\n")
                    distance_dav_boul_dict[f"{distance_metric}"] = dav_boul_score

                    # Calinski and Harabasz score
                    with open(f"{OUTPUT_FOLDER}/cal_har_score_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{distance_metric},{cal_har_score}\n")
                    distance_cal_har_dict[f"{distance_metric}"] = cal_har_score

                    # Cluster length
                    with open(f"{OUTPUT_FOLDER}/cluster_length_comparisons.csv", "a") as csv_file:
                        for j in range(-1, len(set(optics_labels))):
                            length_j = list(optics_labels).count(j)
                            if length_j > 0:
                                csv_file.write(f"{city_mode_name},{embedder_name},{distance_metric},{j},{length_j}\n")

                    # Full file
                    with open(f"{OUTPUT_FOLDER}/full_data.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{distance_metric},{model_time},{sil_score},{dav_boul_score},{cal_har_score}\n")

                    # Markdown Table
                    with open(f"{OUTPUT_FOLDER}/full_data.md", "a") as md_file:
                        md_file.write(f"|{city_mode_name}|{embedder_name}|{distance_metric}|{model_time}|{sil_score}|{dav_boul_score}|{cal_har_score}|\n")


                with Timer(f"Generating scatterplot for clusters with distance metric `{distance_metric}` and embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                    plot_clusters(scatterplot_vectors,
                                filename=f"{OUTPUT_FOLDER}/clusters_{distance_metric}.png",
                                title=f'Clusters Representation ({distance_metric}) for {embedder_name}` ({city_mode_name})',
                                labels=optics_labels)

            # Plotting time bar chart
            with Timer(f"Generating bar chart for time with embedder `{embedder_name}` ({city_mode_name})"):
                Xs = range(len(distance_time_dict.keys()))
                Ys = list(distance_time_dict.values())
                plt.bar(Xs, Ys)
                plt.xticks(Xs, list(distance_time_dict.keys()))
                plt.xlabel("Distance Metric setting")
                plt.ylabel("Time (s)")
                plt.title(f"Time per OPTICS config with embedder `{embedder_name}` ({city_mode_name})")
                plt.savefig(f"{OUTPUT_FOLDER}/time.png")
                plt.close()

            # Plotting silhouette bar chart
            with Timer(f"Generating bar chart for silhouette with embedder `{embedder_name}` ({city_mode_name})"):
                Xs = range(len(distance_silhouette_dict.keys()))
                Ys = list(distance_silhouette_dict.values())
                plt.bar(Xs, Ys)
                plt.xticks(Xs, list(distance_silhouette_dict.keys()))
                plt.xlabel("Distance Metric setting")
                plt.ylabel("Silhouette")
                plt.title(f"Silhouette per OPTICS config with embedder `{embedder_name}` ({city_mode_name})")
                plt.savefig(f"{OUTPUT_FOLDER}/silhouette.png")
                plt.close()

            # Plotting Calinski and Harabasz bar chart
            with Timer(f"Generating bar chart for Calinski and Harabasz with embedder `{embedder_name}` ({city_mode_name})"):
                Xs = range(len(distance_cal_har_dict.keys()))
                Ys = list(distance_cal_har_dict.values())
                plt.bar(Xs, Ys)
                plt.xticks(Xs, list(distance_cal_har_dict.keys()))
                plt.xlabel("Distance Metric setting")
                plt.ylabel("Calinksi and Harabasz score")
                plt.title(f"CaH score per OPTICS config with embedder `{embedder_name}` ({city_mode_name})")
                plt.savefig(f"{OUTPUT_FOLDER}/cal_har.png")
                plt.close()

            # Plotting Davies-Bouldin score bar chart
            with Timer(f"Generating bar chart for Davies-Bouldin score with embedder `{embedder_name}` ({city_mode_name})"):
                Xs = range(len(distance_dav_boul_dict.keys()))
                Ys = list(distance_dav_boul_dict.values())
                plt.bar(Xs, Ys)
                plt.xticks(Xs, list(distance_dav_boul_dict.keys()))
                plt.xlabel("Distance Metric setting")
                plt.ylabel("Davies-Bouldin score")
                plt.title(f"D-B score per OPTICS config with embedder `{embedder_name}` ({city_mode_name})")
                plt.savefig(f"{OUTPUT_FOLDER}/dav_boul.png")
                plt.close()
