"""
Script for easy running of k-means tests in order to fetch information
from the stored data.
"""

import logging
import random

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score

from caupo.preprocessing import preprocess_v1
from caupo.utils import Timer, get_text_from_all_tweets

# Instantiate logger
logger = logging.getLogger("caupo")

with Timer("Main script runtime"):
    # Get all tweets
    with Timer("Getting tweets' text from database"):
        corpus = get_text_from_all_tweets()
    logger.info("Amount of tweets: %s", len(corpus))

    # Normalize tweets
    with Timer("Normalizing tweets' text"):
        preprocessed_corpus = preprocess_v1(corpus)

    # Get rid of duplicate processed tweets (this should take care of duplicate, spammy tweets)
    with Timer("Removing duplicate tweets (bot protection)"):
        clean_corpus = list(set(preprocessed_corpus))
    logger.info("Amount of clean tweets: %s", len(clean_corpus))

    sample_tweet = random.choice(clean_corpus)
    logger.info("Clean tweet example: %s", sample_tweet)

    # TODO: Use a better vectorizer, try different vectorizers
    # TODO: Use prebuild doc2vec
    # Vectorize: Doc2Vec, CountVectorizer, SentenceBert, FastText, un embedding de politica ya prehecho
    with Timer("Vectorizing tweets with Doc2Vec"):
        d2v_model = Doc2Vec([TaggedDocument(doc.split(), [i]) for i, doc in enumerate(clean_corpus)],
                        vector_size=200, window=3, min_count=2, workers=2)
        d2v_vectors = [d2v_model.infer_vector(doc.split()) for doc in clean_corpus]

    BERT_MODEL_NAME = 'paraphrase-xlm-r-multilingual-v1'
    with Timer(f"Vectorizing tweets with BERT (multilingual model, {BERT_MODEL_NAME})"):
        bert_model = SentenceTransformer(BERT_MODEL_NAME, device="cpu")
        bert_vectors = bert_model.encode(clean_corpus)

    embedding_vectors = {
        "Trained Doc2Vec": d2v_vectors,
        "Pretrained Multilingual BERT": bert_vectors,
    }

    for embedding, vectors in embedding_vectors.items():
        logger.info("*** Test with embedding=`%s` ***", embedding)

        # Find clusters
        # Kmeans, un kmeans puyao, affinity propagation, dbscan
        # TODO: Try other clustering algorithms
        ks_inertias = {}
        ks_sils = {}
        MAX_K = 3
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

                logger.info("Rebuilding cluster with original phrases for k=%s", k_clusters)
                clusters_from_corpus = {}
                clusters_from_original_corpus = {}
                for i, phrase in enumerate(clean_corpus):
                    label = km_labels[i]
                    if label not in clusters_from_corpus:
                        clusters_from_corpus[label] = []
                        clusters_from_original_corpus[label] = []
                    clusters_from_corpus[label].append(phrase)
                    clusters_from_original_corpus[label].append(corpus[i])

                # Rebuild each cluster and find topics
                for k, cluster in clusters_from_corpus.items():
                    SHOWCASE_AMOUNT = 15
                    print()
                    print(f"Cluster {k} (size: {len(cluster)} tweets)")
                    print(f"First {SHOWCASE_AMOUNT} tweets:")
                    for phrase in clusters_from_original_corpus[k][:SHOWCASE_AMOUNT]:
                        print(f"(*) {phrase}")
                    print()

                    #? Topic modelling (with LDA, NMF, PLSI)
                    #? src: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

                    # Extract 10 topics and get 10 top words from each one
                    N_TOPICS = 10
                    N_TOP_WORDS = 10
                    N_FEATURES = 1000

                    # Use tf-idf features for NMF.
                    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=N_FEATURES)
                    tfidf = tfidf_vectorizer.fit_transform(cluster)

                    # Use tf (raw term count) features for LDA.
                    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=N_FEATURES)
                    tf = tf_vectorizer.fit_transform(cluster)

                    # LDA
                    lda = LatentDirichletAllocation(n_components=N_TOPICS, max_iter=20, learning_method='online',
                                                    learning_offset=50.0)

                    # NMF (Frobenius norm)
                    nmf_frobenius = NMF(n_components=N_TOPICS, max_iter=1000, alpha=0.1, l1_ratio=.5)

                    # NMF (generalized Kullback-Leibler divergence), equivalent to Probabilistic Latent Semantic Indexing
                    nmf_plsi = NMF(n_components=N_TOPICS, beta_loss='kullback-leibler', solver='mu', max_iter=1000,
                                alpha=0.1, l1_ratio=0.5)

                    topic_models = {
                        'LDA': lda,
                        'NMF (Frobenius)': nmf_frobenius,
                        'NMF (Kullback-Leibler) / PLSI': nmf_plsi,
                    }

                    for name, model in topic_models.items():
                        print(f"Obtaining { N_TOPICS } topics with { name }:")
                        # Fit model
                        model.fit(tf)

                        # Obtain and print first N terms for each topic
                        feature_names = tf_vectorizer.get_feature_names()
                        topics = model.components_
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

    # TODO: Measure topic modelling coherence
