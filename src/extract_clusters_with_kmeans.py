import random
import logging

from itertools import filterfalse
from functools import partial

from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from utils import Timer, get_text_from_all_tweets, remove_accents

with Timer("Script preparation"):
    # Install nltk data, if needed
    nltk_download('stopwords')
    nltk_download('punkt')

    # Load up stopwords
    stop_words = set(stopwords.words('spanish'))

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

with Timer("Main script runtime"):
    # Get all tweets
    with Timer("Getting tweets' text from database"):
        tweets = get_text_from_all_tweets()
        corpus = [t["full_text"] for t in tweets]

    logger.info("Amount of tweets: %s", len(corpus))

    # Normalize tweets
    with Timer("Normalizing tweets' text"):
        logger.info("Lowering case")
        lowercase_corpus = map(lambda x: x.lower(), corpus)

        logger.info("Removing accents")
        unaccented_corpus = map(remove_accents, lowercase_corpus)

        logger.info("Splitting each tweet into words")
        splitted_corpus = map(word_tokenize, unaccented_corpus)

        logger.info("Removing punctuation")
        alphanumeric_corpus = map(partial(filter, lambda x: x.isalpha()), splitted_corpus)

        logger.info("Removing stopwords")
        clean_corpus = map(partial(filterfalse, lambda x: x in stop_words), alphanumeric_corpus)
        corpus_list = list(map(list, clean_corpus))
        final_corpus = list(map(" ".join, corpus_list))

    logger.info("Clean tweet example: %s", random.choice(final_corpus))

    # TODO: Use a better vectorizer
    # Vectorize
    with Timer("Vectorizing tweets"):
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(final_corpus)

    # Find clusters
    ks_inertias = {}
    for k_clusters in range(2, 6):
        with Timer(f"Finding clusters with k={k_clusters}"):
            km = KMeans(n_clusters=k_clusters)
            km.fit(vectors)
            logger.info("Inertia with k=%s: %s", k_clusters, km.inertia_)
            ks_inertias[k_clusters] = km.inertia_

        # TODO: Obtain top 10 terms through tf-idf on each cluster, maybe
        print("Top terms per cluster:")
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        for i in range(k_clusters):
            top_ten_words = [terms[ind] for ind in order_centroids[i, :10]]
            print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))
        print()

    min_inertia = sorted(ks_inertias.items(), key=lambda x: x[1])[0]
    logger.info("Minimum inertia achieved with k=%s (inertia: %s)", min_inertia[0], min_inertia[1])


    # TODO: Análisis de sentimiento
