"""
Script for easy running of k-means tests in order to fetch information
from the stored data.
"""

import logging
import random
from functools import partial
from itertools import filterfalse

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

from utils import Timer, get_text_from_all_tweets, remove_accents, remove_emoji

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

        logger.info("Removing emoji")
        no_emoji_corpus = map(remove_emoji, unaccented_corpus)

        logger.info("Splitting each tweet into words")
        splitted_corpus = map(word_tokenize, no_emoji_corpus)

        logger.info("Removing punctuation and digits")
        alphanumeric_corpus = map(partial(filter, lambda x: x.isalpha()), splitted_corpus)

        logger.info("Removing stopwords")
        clean_corpus = map(partial(filterfalse, lambda x: x in stop_words), alphanumeric_corpus)
        corpus_list = list(map(list, clean_corpus))
        final_corpus = list(map(" ".join, corpus_list))

    sample_tweet_index = random.randrange(0, len(final_corpus))
    logger.info("Original tweet example: %s", corpus[sample_tweet_index])
    logger.info("Clean tweet example: %s", final_corpus[sample_tweet_index])

    # TODO: Use a better vectorizer, try different vectorizers
    # Vectorize
    with Timer("Vectorizing tweets"):
        model = Doc2Vec([TaggedDocument(doc.split(), [i]) for i, doc in enumerate(final_corpus)],
                        vector_size=200, window=3, min_count=2, workers=2)
        vectors = [model.infer_vector(doc.split()) for doc in final_corpus]

    # Find clusters
    # TODO: Try other clustering algorithms
    ks_inertias = {}
    ks_sils = {}
    MAX_K = 5
    for k_clusters in range(2, MAX_K):
        with Timer(f"Finding clusters with k={k_clusters}"):
            km = KMeans(n_clusters=k_clusters)
            km_result = km.fit(vectors)
            km_labels = km_result.labels_
            logger.info("Inertia with k=%s: %s", k_clusters, km.inertia_)
            ks_inertias[k_clusters] = km.inertia_
            sil_score = silhouette_score(vectors, km_labels, metric='euclidean')
            logger.info("Silhouete score with k=%s: %s", k_clusters, sil_score)
            ks_sils[k_clusters] = sil_score

            # TODO: Obtain top 10 terms through tf-idf on each cluster, maybe
            # TODO: Use `corpus` for showcasing, `final_corpus` for tfidf
            #! BUG: Top terms not being calculated properly
            logger.info("Rebuilding cluster with original phrases for k=%s", k_clusters)
            clusters_from_corpus = {}
            for i, phrase in enumerate(final_corpus):
                label = km_labels[i]
                if label not in clusters_from_corpus:
                    clusters_from_corpus[label] = []
                clusters_from_corpus[label].append(phrase)

            #? Topic modelling with LDA
            # TODO: Test other modelling algorithms
            #? src: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
            for k, cluster in clusters_from_corpus.items():
                SHOWCASE_AMOUNT = 15
                print()
                print(f"Cluster {k}")
                print(f"First {SHOWCASE_AMOUNT} tweets:")
                for phrase in cluster[:SHOWCASE_AMOUNT]:
                    print(f"(*) {phrase}")
                print()

                # Use tf (raw term count) features for LDA.
                tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words=stop_words)
                tf = tf_vectorizer.fit_transform(cluster)

                # Extract 10 clusters / topics
                lda = LatentDirichletAllocation(n_components=10, max_iter=10, learning_method='online',
                                                learning_offset=50.0)
                lda.fit(tf)

                # Obtain and print first N terms for each topic
                feature_names = tf_vectorizer.get_feature_names()
                N_TOP_WORDS = 10
                topics = lda.components_
                for i_topic, topic in enumerate(topics):
                    top_features_ind = topic.argsort()[:-N_TOP_WORDS - 1:-1]
                    top_features = [feature_names[i] for i in top_features_ind]
                    weights = topic[top_features_ind]
                    print(f"Topic { i_topic }: { top_features }")
                    print(f"Weights {i_topic}: { weights }")
                print()


    min_inertia = sorted(ks_inertias.items(), key=lambda x: x[1])[0]
    logger.info("Minimum inertia achieved with k=%s (inertia: %s)", min_inertia[0], min_inertia[1])

    max_silhouette = sorted(ks_sils.items(), key=lambda x: x[1], reverse=True)[0]
    logger.info("Maximum silhouette score achieved with k=%s (silhouette score: %s)",
                max_silhouette[0], max_silhouette[1])

    # TODO: Análisis de sentimiento para ver la polaridad en cada cluster (opinion mining)
    # TODO: En cada topic, también ver polaridad

    # TODO: Mejorar eliminación de cuentas basura
    # TODO: Eliminar URLs y @s
