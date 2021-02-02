import logging

from itertools import filterfalse
from functools import partial

from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

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
    clean_corpus = map(partial(filterfalse, lambda x: x in stop_words), alphanumeric_corpus)
    corpus_list = list(map(list, clean_corpus))
    final_corpus = list(map(" ".join, corpus_list))

logger.info("Clean tweet example: %s", final_corpus[0])

# TODO: Use a better vectorizer
# Vectorize
with Timer("Vectorizing tweets"):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(final_corpus)

# Find clusters
k_clusters = 2
with Timer(f"Finding clusters with k={k_clusters}"):
    km = KMeans(n_clusters=k_clusters)
    km.fit(vectors)

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(number_of_clusters):
    top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
    print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))

# TODO: K-means para hallar los clusters de documentos
# TODO: Aplicar tf-idf a cada cluster
# TODO: Ver los 10 términos más importantes
# TODO: Análisis de sentimiento
