"""
Module that runs a series of experiments and plots in order to analyse
possible values of `eps` to be used in DBSCAN for each embedder

src: https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
"""

import logging
import os
import time
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

from caupo.embeddings import get_embedder_functions
from caupo.preprocessing import preprocess_corpus
from caupo.utils import (Timer, get_city_modes, get_text_from_all_tweets,
                         plot_clusters)

mpl.use('Agg')
sns.set()

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
OUTPUT_FOLDER = f"outputs/measure_possible_eps_values_for_embeddings/{ timestamp }"
os.makedirs(OUTPUT_FOLDER)

# Add file handler to the logger
file_handler = logging.FileHandler(f'{OUTPUT_FOLDER}/measure_possible_eps_values_for_embeddings.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


with Timer("Main script runtime"):
    city_modes = get_city_modes().items()
    embedders = get_embedder_functions().items()

    city_embedder_time_dict = {}
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

            embedder_time_dict = {}
            for embedder_name, embedder_function in embedders:
                with Timer(f"Getting vectors with embedder `{embedder_name}`"):
                    t0 = time.time()
                    vectors = embedder_function(clean_corpus)
                    t1 = time.time()

                DISTANCE_METRICS = ["euclidean", "cosine"]

                for distance_metric in DISTANCE_METRICS:

                    with Timer(f"Getting nearest neighbors using distance `{distance_metric}` with embedder `{embedder_name}`"):
                        neigh = NearestNeighbors(n_neighbors=2, metric=distance_metric)
                        neighbors_data = neigh.fit(vectors)
                        distances, _ = neighbors_data.kneighbors(vectors)

                    with Timer(f"Sorting and plotting neighbors using distance `{distance_metric}` with embedder `{embedder_name}`"):
                        distances = np.sort(distances, axis=0)
                        distances = distances[:,1]
                        plt.plot(distances)
                        plt.title(f"Distance to nearest neighbor ({embedder_name})")
                        plt.xlabel("Elements (sorted by closest - longest distance)")
                        plt.ylabel(f"Shortest distance (`{distance_metric}`)")
                        plt.savefig(f"{OUTPUT_FOLDER}/neighbors_{embedder_name}_{distance_metric}.png")
                        plt.close()
