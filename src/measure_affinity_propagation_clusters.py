"""
Module that runs a series of tests to measure clustering of tweets
with Affinity Propagation, using different available word embeddings
"""

import time
import logging
import os
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
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
                md_file.write("|City mode|Embedder|Time of Clustering|Silhouette score|Davies-Bouldin score|Calinski and Harabasz score|\n")
                md_file.write("|---------|--------|------------------|----------------|--------------------|---------------------------|\n")

            with Timer(f"Finding clusters with embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                mean_shift = AffinityPropagation()
                t0 = time.time()
                mean_shift_result = mean_shift.fit(vectors)
                t1 = time.time()

            with Timer(f"Getting metrics with embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                model_time = t1 - t0
                mean_shift_labels = mean_shift_result.labels_
                logger.info("Mean Shift generated `%s` cluster(s)", len({i for i in mean_shift_labels if i != -1}))

                logger.debug("Calculating silhouette score")
                sil_score = silhouette_score(vectors, mean_shift_labels)
                logger.info("Silhouete score: %s", sil_score)

                logger.debug("Calculating Davies-Boulding score")
                dav_boul_score = davies_bouldin_score(vectors, mean_shift_labels)
                logger.info("Davies-Boulding score: %s", dav_boul_score)

                logger.debug("Calculating Calinski & Harabasz score")
                cal_har_score = calinski_harabasz_score(vectors, mean_shift_labels)
                logger.info("Calinski & Harabasz score: %s", cal_har_score)

            with Timer(f"Storing results with embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                # Time
                with open(f"{OUTPUT_FOLDER}/time_comparisons.csv", "a") as csv_file:
                    csv_file.write(f"{city_mode_name},{embedder_name},{model_time}\n")

                # Silhouette
                with open(f"{OUTPUT_FOLDER}/silhouette_comparisons.csv", "a") as csv_file:
                    csv_file.write(f"{city_mode_name},{embedder_name},{sil_score}\n")

                # Davies-Bouldin score
                with open(f"{OUTPUT_FOLDER}/dav_boul_score_comparisons.csv", "a") as csv_file:
                    csv_file.write(f"{city_mode_name},{embedder_name},{dav_boul_score}\n")

                # Calinski and Harabasz score
                with open(f"{OUTPUT_FOLDER}/cal_har_score_comparisons.csv", "a") as csv_file:
                    csv_file.write(f"{city_mode_name},{embedder_name},{cal_har_score}\n")

                # Cluster length
                with open(f"{OUTPUT_FOLDER}/cluster_length_comparisons.csv", "a") as csv_file:
                    for j in range(-1, len(set(mean_shift_labels))):
                        length_j = list(mean_shift_labels).count(j)
                        if length_j > 0:
                            csv_file.write(f"{city_mode_name},{embedder_name},{j},{length_j}\n")

                # Full file
                with open(f"{OUTPUT_FOLDER}/full_data.csv", "a") as csv_file:
                    csv_file.write(f"{city_mode_name},{embedder_name},{model_time},{sil_score},{dav_boul_score},{cal_har_score}\n")

                # Markdown Table
                with open(f"{OUTPUT_FOLDER}/full_data.md", "a") as md_file:
                    md_file.write(f"|{city_mode_name}|{embedder_name}|{model_time}|{sil_score}|{dav_boul_score}|{cal_har_score}|\n")


            with Timer(f"Generating scatterplot for clusters with embedder `{embedder_name}` for city mode `{city_mode_name}`"):
                plot_clusters(scatterplot_vectors,
                            filename=f"{OUTPUT_FOLDER}/clusters.png",
                            title=f'Clusters Representation for {embedder_name}` ({city_mode_name})',
                            labels=mean_shift_labels)

            # TODO: In this case these should be embedder-wise charts maybe
