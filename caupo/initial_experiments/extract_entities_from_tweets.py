"""
Module that performs named entity recognition / extraction (NER) on the text of stored tweets, using Spacy's models,
and generating word clouds to visualize the result
"""

import logging
import os
from datetime import datetime

from wordcloud import WordCloud

from caupo.preprocessing import get_stopwords, nlp
from caupo.utils import Timer, get_city_modes, get_text_from_all_tweets

# Instantiate logger
logger = logging.getLogger("caupo")

# Creating folder for output
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
BASE_OUTPUT_FOLDER = f"outputs/extract_entities_from_tweets/{ timestamp }"
os.makedirs(BASE_OUTPUT_FOLDER)

with Timer("Main script runtime"):
    city_modes = get_city_modes().items()
    stop_words = get_stopwords()

    for city_mode_name, city_mode_tag in city_modes:
        with Timer(f"Running script for city {city_mode_name}"):
            with Timer("Getting tweets' text from database"):
                corpus = get_text_from_all_tweets(city=city_mode_tag)
            logger.info("Amount of tweets: %s", len(corpus))

            # Get rid of duplicate processed tweets (this should take care of duplicate, spammy tweets)
            with Timer("Removing duplicate tweets (bot protection)"):
                clean_corpus = list(set(corpus))
            logger.info("Amount of clean tweets: %s", len(clean_corpus))

            types_entities = {}
            with Timer("Obtaining entities for each tweet and ent type"):
                processed_corpus = [nlp(x) for x in clean_corpus]
                entities = [[(X.text, X.label_) for X in doc.ents] for doc in processed_corpus]
                entity_types_list = [{X.label_ for X in doc.ents} for doc in processed_corpus]
                entity_types_reduced = set().union(*entity_types_list)

                for type in entity_types_reduced:
                    type_entities_lists = [[entity for entity, label in ent_list if label == type] for ent_list in entities]
                    entities_list = []
                    for type_entity_list in type_entities_lists:
                        entities_list.extend(type_entity_list)
                    types_entities[type] = entities_list

            with Timer("Generating word cloud from texts for each type"):
                for type, entities in types_entities.items():
                    logger.debug("Current type: %s", type)
                    wcloud = WordCloud(width=800, height=400, max_words=150, stopwords=stop_words).generate(" ".join(entities))
                    wcloud.to_file(f"{BASE_OUTPUT_FOLDER}/entities_cloud_{city_mode_name}_{type}.png")
