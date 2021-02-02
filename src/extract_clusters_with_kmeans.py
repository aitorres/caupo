import logging

from functools import partial

from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from utils import Timer, get_text_from_all_tweets

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

# Get all tweets
with Timer("Getting tweets' text from database"):
    tweets = get_text_from_all_tweets()
    corpus = [t["full_text"] for t in tweets]

logger.info("Amount of tweets: %s", len(corpus))

# Normalize tweets
with Timer("Normalizing tweets' text"):
    logger.info("Splitting each tweet into words")
    splitted_corpus = map(word_tokenize, corpus)

    logger.info("Removing punctuation")
    alphanumeric_corpus = map(partial(filter, lambda x: x.isalpha()), splitted_corpus)

    logger.info("Removing stopwords")
    clean_corpus = list(map(partial(filter, lambda x: x in stop_words), alphanumeric_corpus))
    final_corpus = list(map(list, clean_corpus))

logger.info("Clean tweet example: %s", final_corpus[0])

# TODO: Vectorizar
# TODO: K-means para hallar los clusters de documentos
# TODO: Aplicar tf-idf a cada cluster
# TODO: Ver los 10 términos más importantes
# TODO: Análisis de sentimiento
