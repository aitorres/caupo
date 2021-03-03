"""
Module that runs a series of tests to get measurements from the
different available word embeddings.
"""

import time
import logging
import os
import random
from datetime import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

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

# Add file handler to the logger
file_handler = logging.FileHandler('measure_embeddings.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Creating folder for output
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
OUTPUT_FOLDER = f"outputs/measure_embeddings/{ timestamp }"
os.makedirs(OUTPUT_FOLDER)

# Add headers to MD file
with open(f"{OUTPUT_FOLDER}/full_data.md", "a") as md_file:
    md_file.write(f"# Results ( {timestamp} )\n\n")
    md_file.write("|City Mode|Embedder|Max 2-Norm|Min 2-Norm|Avg 2-Norm|\n")
    md_file.write("|---|---|---|---|---|\n")

with Timer("Main script runtime"):
    city_modes = {
        'Caracas': 'Caracas',
        'All cities': None,
    }
    embedders = get_embedder_functions().items()

    city_embedder_time_dict = {}
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
            logger.info("Amount of clean tweets: %s", len(corpus))

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

                try:
                    with Timer(f"Generating scatterplot for vector representations with embedder `{embedder_name}`"):
                        data_sample = random.sample(list(vectors), 1000)
                        fit = UMAP()
                        scatterplot_data = fit.fit_transform(data_sample)
                        plt.scatter(scatterplot_data[:,0], scatterplot_data[:,1])
                        plt.title(f'2-Dim Representation of `{embedder_name}` ({city_mode_name})')
                        plt.savefig(f"{OUTPUT_FOLDER}/{embedder_name}-{city_mode_name}-scatter.png")
                        plt.close()
                except ValueError:
                    logger.error("Value Error trying to generate scatterplot for vector reps with embedder `%s`", embedder_name)

            city_embedder_time_dict[city_mode_name] = embedder_time_dict

    with Timer("Generating and storing time plot"):
        # src: https://matplotlib.org/3.3.4/gallery/lines_bars_and_markers/barchart.html
        labels = [name for name, _ in embedders]
        caracas_times = [t for _, t in city_embedder_time_dict['Caracas'].items()]
        all_cities_times = [t for _, t in city_embedder_time_dict['All cities'].items()]
        x = np.arange(labels)

        WIDTH = 0.35
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - WIDTH/2, caracas_times, WIDTH, label='Caracas')
        rects2 = ax.bar(x + WIDTH/2, all_cities_times, WIDTH, label='All cities')

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
        autolabel(rects2)

        fig.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/time_plot.png")
        plt.close()
