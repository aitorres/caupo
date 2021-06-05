"""
Auxiliary module to add wrappers around topic modelling algorithms in order to
reuse them for several different tasks

ref: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
"""

import logging
from typing import Callable, Dict, List, Tuple, Union

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from caupo.preprocessing import get_stopwords

logger = logging.getLogger("caupo")
stopwords = get_stopwords()


def _get_tfidf_vectorizer() -> TfidfVectorizer:
    """Instantiates and returns a Tfidf Vectorizer instance"""

    return TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=1000,
        stop_words=stopwords
    )


def _get_tf_vectorizer() -> CountVectorizer:
    """Instantiates and returns a Term Frequency (Count) Vectorizer instance"""

    return CountVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=1000,
        stop_words=stopwords
    )


def nmf_frobenius_topics(documents: List[str]) -> Tuple[NMF, List[str]]:
    """
    Given a series of documents, performs NMF-based
    topic modelling with Frobenius norm
    """

    tfidf_vectorizer = _get_tfidf_vectorizer()
    tfidf = tfidf_vectorizer.fit_transform(documents)
    feature_names = tfidf_vectorizer.get_feature_names()
    nmf = NMF(
        n_components=3,
        init=None,
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=1000
    ).fit(tfidf)
    return nmf, feature_names


def plsi_topics(documents: List[str]) -> Tuple[NMF, List[str]]:
    """
    Given a series of documents, performs NMF-based topic modelling
    with the generalized Kullback-Leibler divergence as norm, which
    makes the NMF model equivalent to Probabilistic Latent Semantic Indexing
    """

    tfidf_vectorizer = _get_tfidf_vectorizer()
    tfidf = tfidf_vectorizer.fit_transform(documents)
    feature_names = tfidf_vectorizer.get_feature_names()
    nmf = NMF(
        n_components=3,
        init=None,
        beta_loss='kullback-leibler',
        solver='mu',
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=1000
    ).fit(tfidf)
    return nmf, feature_names


def lda_topics(documents: List[str]) -> Tuple[LatentDirichletAllocation, List[str]]:
    """
    Given a series of documents, performs LDA-based topic modelling
    with the Latent Dirichlet Allocation algorithm.
    """

    tf_vectorizer = _get_tf_vectorizer()
    tf = tf_vectorizer.fit_transform(documents)
    feature_names = tf_vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(
        n_components=3,
        max_iter=100,
        learning_method='online',
        learning_offset=50.0,
    ).fit(tf)
    return lda, feature_names


def get_topic_models() -> Dict[str, Callable[[List[str]], Tuple[Union[NMF, LatentDirichletAllocation], List[str]]]]:
    """Returns a dictionary that contains the name and model instance for all topic models available"""

    return {
        'LDA': lda_topics,
        'PLSI': plsi_topics,
        'NMF': nmf_frobenius_topics,
    }


def main():
    """Runs a small test client to test the implementation of the algorithms"""

    documents = [
        "hola esto es un documento",
        "esto es otro",
        "documento uno dos tres",
        "podriamos escribir un tratado con estos documentos",
        "aqui te va el quinto documento",
        "no se que es un articulo otro",
    ]
    n_top_words = 5

    for topic_model_name, topic_model in get_topic_models().items():
        logger.info("Testing %s", topic_model_name)
        model, feature_names = topic_model(documents)
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            logger.info("Topic %s has top words %s with weights %s", topic_idx, top_features, weights)


if __name__ == "__main__":
    main()
