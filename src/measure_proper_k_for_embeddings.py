"""
Module that runs a series of tests to get measurements of proper
K values for different available word embeddings
"""

import time
import logging
import os
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from embeddings import get_embedder_functions
from preprocessing import preprocess_corpus
from utils import get_text_from_all_tweets, Timer

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
BASE_OUTPUT_FOLDER = f"outputs/measure_proper_k_for_embeddings/{ timestamp }"
os.makedirs(BASE_OUTPUT_FOLDER)

# Add file handler to the logger
file_handler = logging.FileHandler(f'{BASE_OUTPUT_FOLDER}/measure_proper_k_for_embeddings.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


with Timer("Main script runtime"):
    city_modes = {
        'Caracas': 'Caracas',
        'All cities': None,
    }
    embedders = get_embedder_functions().items()

    for city_mode_name, city_mode_tag in city_modes.items():
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
            with Timer(f"Getting vectors with embedder `{embedder_name}`"):
                vectors = embedder_function(clean_corpus)

            with Timer(f"Getting reduced vectors (2D) for scatterplot"):
                pca_fit = PCA(n_components=2)
                scatterplot_data = pca_fit.fit_transform(vectors)

            OUTPUT_FOLDER = f"{BASE_OUTPUT_FOLDER}/{city_mode_name}/{embedder_name}"
            os.makedirs(OUTPUT_FOLDER)
            with open(f"{OUTPUT_FOLDER}/full_data.md", "a") as md_file:
                md_file.write(f"|City mode|Embedder|Number of Clusters (K)|Time of Clustering|Silhouette|Inertia|\n")
                md_file.write(f"|---------|--------|----------------------|------------------|----------|-------|\n")

            MAX_K = 8
            ALL_KS = list(range(2, MAX_K + 1))
            logger.info("Setting max K =`%s`", MAX_K)
            k_time_dict = {}
            k_silhouette_dict = {}
            k_inertia_dict = {}
            for k_clusters in ALL_KS:
                with Timer(f"Finding clusters with k=`{k_clusters}` and embedder `{embedder_name}`"):
                    t0 = time.time()
                    km = KMeans(n_clusters=k_clusters)
                    t1 = time.time()
                    km_result = km.fit(vectors)

                with Timer(f"Getting metrics with k=`{k_clusters}` and embedder `{embedder_name}`"):
                    model_time = t1 - t0
                    km_labels = km_result.labels_
                    inertia = km.inertia_
                    sil_score = silhouette_score(vectors, km_labels, metric='euclidean')
                    logger.info("Inertia with k=%s: %s", k_clusters, km.inertia_)
                    logger.info("Silhouete score with k=%s: %s", k_clusters, sil_score)

                with Timer(f"Storing results with k=`{k_clusters}` and embedder `{embedder_name}`"):
                    # Time
                    with open(f"{OUTPUT_FOLDER}/time_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{k_clusters},{model_time}\n")
                    k_time_dict[k_clusters] = model_time

                    # Silhouette
                    with open(f"{OUTPUT_FOLDER}/silhouette_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{k_clusters},{sil_score}\n")
                    k_silhouette_dict[k_clusters] = sil_score

                    # Inertia
                    with open(f"{OUTPUT_FOLDER}/inertia_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{k_clusters},{inertia}\n")
                    k_inertia_dict[k_clusters] = inertia

                    # Cluster length
                    with open(f"{OUTPUT_FOLDER}/cluster_length_comparisons.csv", "a") as csv_file:
                        for j in range(0, k_clusters):
                            length_j = list(km_labels).count(j)
                            csv_file.write(f"{city_mode_name},{embedder_name},{k_clusters},{j},{length_j}\n")

                    # Full file
                    with open(f"{OUTPUT_FOLDER}/full_data.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{k_clusters},{model_time},{sil_score},{inertia}\n")

                    # Markdown Table
                    with open(f"{OUTPUT_FOLDER}/full_data.md", "a") as md_file:
                        md_file.write(f"|{city_mode_name}|{embedder_name}|{k_clusters}|{model_time}|{sil_score}|{inertia}|\n")

                with Timer(f"Generating scatterplot for clusters  k=`{k_clusters}` and embedder `{embedder_name}`"):
                    plt.scatter(scatterplot_data[:,0], scatterplot_data[:,1], c=km_labels, s=12)
                    plt.title(f'Clusters Representation (k={k_clusters}) for `{embedder_name}` ({city_mode_name})')
                    plt.savefig(f"{OUTPUT_FOLDER}/clusters_{k_clusters}.png")
                    plt.close()

            # Plotting time for each K
            with Timer(f"Generating bar chart for time with embedder `{embedder_name}` ({city_mode_name})"):
                Xs = ALL_KS
                Ys = [k_time_dict[i] for i in Xs]
                plt.bar(Xs, Ys)
                plt.xticks(Xs, [f"K={k}" for k in Xs])
                plt.xlabel("Size of clusters (K)")
                plt.ylabel("Time (s)")
                plt.title(f"Time per K with embedder `{embedder_name}` ({city_mode_name})")
                plt.savefig(f"{OUTPUT_FOLDER}/time.png")
                plt.close()

            # Plotting inertia scatterplot with trend line
            with Timer(f"Generating scatterplot for inertias with embedder `{embedder_name}` ({city_mode_name})"):
                Xs = ALL_KS
                Ys = [k_inertia_dict[i] for i in Xs]
                plt.plot(Xs, Ys, '-o')
                plt.xlabel("Size of clusters (K)")
                plt.ylabel("Inertia")
                plt.title(f"Inertia per K with embedder `{embedder_name}` ({city_mode_name})")
                plt.savefig(f"{OUTPUT_FOLDER}/inertia.png")
                plt.close()

            # Plotting silhouette scatterplot with trend line
            with Timer(f"Generating scatterplot for silhouette with embedder `{embedder_name}` ({city_mode_name})"):
                Xs = ALL_KS
                Ys = [k_silhouette_dict[i] for i in Xs]
                plt.plot(Xs, Ys, '-o')
                plt.xlabel("Size of clusters (K)")
                plt.ylabel("Silhouette")
                plt.title(f"Silhouette per K with embedder `{embedder_name}` ({city_mode_name})")
                plt.savefig(f"{OUTPUT_FOLDER}/silhouette.png")
                plt.close()

