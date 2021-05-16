"""
Module that runs a series of tests to measure clustering of tweets
with Mean Shift, using different available word embeddings
"""

import logging
import os
import time
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from caupo.embeddings import get_embedder_functions
from caupo.preprocessing import preprocess_corpus
from caupo.utils import (Timer, get_city_modes, get_text_from_all_tweets,
                         plot_clusters)

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
BASE_OUTPUT_FOLDER = f"outputs/measure_mean_shift_clustering/{ timestamp }"
os.makedirs(BASE_OUTPUT_FOLDER)

# Add file handler to the logger
file_handler = logging.FileHandler(f'{BASE_OUTPUT_FOLDER}/measure_mean_shift_clustering.log')
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
                md_file.write("|City mode|Embedder|Cluster all?|Time of Clustering|Silhouette score|Davies-Bouldin score|Calinski and Harabasz score|\n")
                md_file.write("|---------|--------|------------|------------------|----------------|--------------------|---------------------------|\n")


            cluster_all_values = [True, False]

            clustering_type_time_dict = {}
            clustering_type_silhouette_dict = {}
            clustering_type_dav_boul_dict = {}
            clustering_type_cal_har_dict = {}
            for cluster_all in cluster_all_values:
                logger.info("Starting evaluation with cluster_all=`%s`", cluster_all)

                with Timer(f"Finding clusters with cluster_all=`{cluster_all}` and embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                    mean_shift = MeanShift(cluster_all=cluster_all, n_jobs=-1)
                    t0 = time.time()
                    mean_shift_result = mean_shift.fit(vectors)
                    t1 = time.time()

                with Timer(f"Getting metrics with cluster_all=`{cluster_all}` and embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                    model_time = t1 - t0
                    mean_shift_labels = mean_shift_result.labels_
                    logger.info("Mean Shift generated `%s` cluster(s)", len({i for i in mean_shift_labels if i != -1}))

                    logger.debug("Calculating silhouette score")
                    sil_score = silhouette_score(vectors, mean_shift_labels)
                    logger.info("Silhouete score with cluster_all=`%s`: %s", cluster_all, sil_score)

                    logger.debug("Calculating Davies-Boulding score")
                    dav_boul_score = davies_bouldin_score(vectors, mean_shift_labels)
                    logger.info("Davies-Boulding score with cluster_all=`%s`: %s", cluster_all, dav_boul_score)

                    logger.debug("Calculating Calinski & Harabasz score")
                    cal_har_score = calinski_harabasz_score(vectors, mean_shift_labels)
                    logger.info("Calinski & Harabasz score with cluster_all=`%s`: %s", cluster_all, cal_har_score)

                with Timer(f"Storing results with cluster_all=`{cluster_all}` and embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                    # Time
                    with open(f"{OUTPUT_FOLDER}/time_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{cluster_all},{model_time}\n")
                    clustering_type_time_dict[f"{cluster_all}"] = model_time

                    # Silhouette
                    with open(f"{OUTPUT_FOLDER}/silhouette_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{cluster_all},{sil_score}\n")
                    clustering_type_silhouette_dict[f"{cluster_all}"] = sil_score

                    # Davies-Bouldin score
                    with open(f"{OUTPUT_FOLDER}/dav_boul_score_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{cluster_all},{dav_boul_score}\n")
                    clustering_type_dav_boul_dict[f"{cluster_all}"] = dav_boul_score

                    # Calinski and Harabasz score
                    with open(f"{OUTPUT_FOLDER}/cal_har_score_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{cluster_all},{cal_har_score}\n")
                    clustering_type_cal_har_dict[f"{cluster_all}"] = cal_har_score

                    # Cluster length
                    with open(f"{OUTPUT_FOLDER}/cluster_length_comparisons.csv", "a") as csv_file:
                        for j in range(-1, len(set(mean_shift_labels))):
                            length_j = list(mean_shift_labels).count(j)
                            if length_j > 0:
                                csv_file.write(f"{city_mode_name},{embedder_name},{cluster_all},{j},{length_j}\n")

                    # Full file
                    with open(f"{OUTPUT_FOLDER}/full_data.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{cluster_all},{model_time},{sil_score},{dav_boul_score},{cal_har_score}\n")

                    # Markdown Table
                    with open(f"{OUTPUT_FOLDER}/full_data.md", "a") as md_file:
                        md_file.write(f"|{city_mode_name}|{embedder_name}|{cluster_all}|{model_time}|{sil_score}|{dav_boul_score}|{cal_har_score}|\n")


                with Timer(f"Generating scatterplot for clusters with cluster_all=`{cluster_all}` and embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                    plot_clusters(scatterplot_vectors,
                                filename=f"{OUTPUT_FOLDER}/clusters_{cluster_all}.png",
                                title=f'Clusters Representation ({cluster_all}) for {embedder_name}` ({city_mode_name})',
                                labels=mean_shift_labels)

            # Plotting time bar chart
            with Timer(f"Generating bar chart for time with embedder `{embedder_name}` ({city_mode_name})"):
                Xs = range(len(clustering_type_time_dict.keys()))
                Ys = list(clustering_type_time_dict.values())
                plt.bar(Xs, Ys)
                plt.xticks(Xs, list(clustering_type_time_dict.keys()))
                plt.xlabel("Hard Clustering setting")
                plt.ylabel("Time (s)")
                plt.title(f"Time per Mean Shift config with embedder `{embedder_name}` ({city_mode_name})")
                plt.savefig(f"{OUTPUT_FOLDER}/time.png")
                plt.close()

            # Plotting silhouette bar chart
            with Timer(f"Generating bar chart for silhouette with embedder `{embedder_name}` ({city_mode_name})"):
                Xs = range(len(clustering_type_silhouette_dict.keys()))
                Ys = list(clustering_type_silhouette_dict.values())
                plt.bar(Xs, Ys)
                plt.xticks(Xs, list(clustering_type_silhouette_dict.keys()))
                plt.xlabel("Hard Clustering setting")
                plt.ylabel("Silhouette")
                plt.title(f"Silhouette per Mean Shift config with embedder `{embedder_name}` ({city_mode_name})")
                plt.savefig(f"{OUTPUT_FOLDER}/silhouette.png")
                plt.close()

            # Plotting Calinski and Harabasz bar chart
            with Timer(f"Generating bar chart for Calinski and Harabasz with embedder `{embedder_name}` ({city_mode_name})"):
                Xs = range(len(clustering_type_cal_har_dict.keys()))
                Ys = list(clustering_type_cal_har_dict.values())
                plt.bar(Xs, Ys)
                plt.xticks(Xs, list(clustering_type_cal_har_dict.keys()))
                plt.xlabel("Hard Clustering setting")
                plt.ylabel("Calinksi and Harabasz score")
                plt.title(f"CaH score per Mean Shift config with embedder `{embedder_name}` ({city_mode_name})")
                plt.savefig(f"{OUTPUT_FOLDER}/cal_har.png")
                plt.close()

            # Plotting Davies-Bouldin score bar chart
            with Timer(f"Generating bar chart for Davies-Bouldin score with embedder `{embedder_name}` ({city_mode_name})"):
                Xs = range(len(clustering_type_dav_boul_dict.keys()))
                Ys = list(clustering_type_dav_boul_dict.values())
                plt.bar(Xs, Ys)
                plt.xticks(Xs, list(clustering_type_dav_boul_dict.keys()))
                plt.xlabel("Hard Clustering setting")
                plt.ylabel("Davies-Bouldin score")
                plt.title(f"D-B score per Mean Shift config with embedder `{embedder_name}` ({city_mode_name})")
                plt.savefig(f"{OUTPUT_FOLDER}/dav_boul.png")
                plt.close()
