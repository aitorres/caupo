"""
Module that runs a series of tests to get measurements from the
different available word embeddings.
"""

import logging
import os
import random
import time
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from caupo.embeddings import get_embedder_functions
from caupo.preprocessing import preprocess_corpus
from caupo.utils import (Timer, get_city_modes, get_text_from_all_tweets,
                         plot_clusters)

mpl.use('Agg')

# Instantiate logger
logger = logging.getLogger("caupo")

# Creating folder for output
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
OUTPUT_FOLDER = f"outputs/measure_embeddings/{ timestamp }"
os.makedirs(OUTPUT_FOLDER)


# Add headers to MD file
with open(f"{OUTPUT_FOLDER}/full_data.md", "a") as md_file:
    md_file.write(f"# Results ( {timestamp} )\n\n")
    md_file.write("|City Mode|Embedder|Time (s)|Max 2-Norm|Min 2-Norm|Avg 2-Norm|\n")
    md_file.write("|---|---|---|---|---|---|\n")


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

                with Timer(f"Calculating measures for embedder `{embedder_name}`"):
                    embedder_time = round(t1 - t0, 4)
                    l2_norms = list(map(np.linalg.norm, vectors))
                    min_l2_norm = min(l2_norms)
                    max_l2_norm = max(l2_norms)
                    avg_l2_norm = np.mean(l2_norms)

                with Timer(f"Storing measures for embedder `{embedder_name}`"):
                    # Time
                    with open(f"{OUTPUT_FOLDER}/time_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{embedder_time}\n")
                    embedder_time_dict[embedder_name] = embedder_time

                    # Max Norm
                    with open(f"{OUTPUT_FOLDER}/max_norm_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{max_l2_norm}\n")

                    # Min Norm
                    with open(f"{OUTPUT_FOLDER}/min_norm_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{min_l2_norm}\n")

                    # Avg Norm
                    with open(f"{OUTPUT_FOLDER}/avg_norm_comparisons.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{avg_l2_norm}\n")

                    # Full file
                    with open(f"{OUTPUT_FOLDER}/full_data.csv", "a") as csv_file:
                        csv_file.write(f"{city_mode_name},{embedder_name},{embedder_time},{max_l2_norm},{min_l2_norm},{avg_l2_norm}\n")

                    # Markdown Table
                    with open(f"{OUTPUT_FOLDER}/full_data.md", "a") as md_file:
                        md_file.write(f"|{city_mode_name}|{embedder_name}|{embedder_time}|{max_l2_norm}|{min_l2_norm}|{avg_l2_norm}|\n")

                with Timer(f"Generating scatterplot for vector representations with embedder `{embedder_name}`"):
                    pca_fit = PCA(n_components=2)
                    scatterplot_vectors = pca_fit.fit_transform(vectors)
                    plot_clusters(scatterplot_vectors,
                                filename=f"{OUTPUT_FOLDER}/{embedder_name}-{city_mode_name}-scatter.png",
                                title=f'2-Dim Representation of `{embedder_name}` ({city_mode_name})')

            city_embedder_time_dict[city_mode_name] = embedder_time_dict

    with Timer("Generating and storing time plot"):
        # src: https://matplotlib.org/3.3.4/gallery/lines_bars_and_markers/barchart.html
        labels = [name for name, _ in embedders]
        caracas_times = [t for _, t in city_embedder_time_dict['Caracas'].items()]
        x = np.arange(len(labels))

        WIDTH = 0.30
        fig, ax = plt.subplots()
        rects1 = ax.bar(x, caracas_times, WIDTH, label='Caracas')

        ax.set_ylabel('Time (s)')
        ax.set_title('Time by embedding and dataset')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)

        fig.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/time_plot.png")
        plt.close()
