"""
Auxiliary module to add implementations of several word embeddings
in order to test them for different tasks.
"""

import logging
import os
from typing import Callable, Dict, List

import fasttext
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("caupo")


def scale_vectors(vectors: List[float], scale: bool = True) -> List[float]:
    """
    Given a list of vector representations, returns an equivalent list of
    vector representations that have been scaled and standarized (i.e. minus mean
    and divided by std).
    """

    # In case we don't _really_ want to scale
    if not scale:
        return vectors

    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(vectors)

    return scaled_vectors


def doc2vec_embedder(corpus: List[str]) -> List[float]:
    """
    Given a corpus of texts, returns an embedding (representation
    of such texts) using a fine-tuned Doc2Vec embedder.

    ref: https://radimrehurek.com/gensim/models/doc2vec.html
    """

    logger.debug("Instantiating Doc2Vec model")
    tagged_documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(corpus)]
    model = Doc2Vec(tagged_documents, vector_size=100, window=5, min_count=3, workers=16)

    def embedder(documents: List[str]) -> List[float]:
        """Generates an embedding using a Doc2Vec"""
        return scale_vectors([model.infer_vector(doc.split()) for doc in documents])

    return embedder


def bow_embedder(corpus: List[str]) -> Callable[[List[str]], List[float]]:
    """
    Given a corpus of texts, returns a callable that generates embeddings (representation
    of such texts) using a Count Vectorizer for Bag of Words.

    ref: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    """

    logger.debug("Instantiating CountVectorizer model")
    model = CountVectorizer(min_df=3, max_df=0.85).fit(corpus)

    def embedder(documents: List[str]) -> List[float]:
        """Generates an embedding using a Count Vectorizer for Bag of Words"""
        return scale_vectors(model.transform(documents).toarray())

    return embedder


def fasttext_embedder(corpus: List[str], model_type: str = 'cbow') -> Callable[[List[str]], List[float]]:
    """
    Given a corpus of texts and optionally a model type, returns an embedding
    (representation of such texts) using Fast Text as the embedder.

    ref: https://radimrehurek.com/gensim/models/fasttext.html
    """

    # Store corpus temporarily on disk
    corpus_path = "_corpus.txt"
    with open(corpus_path, "w") as temp_file:
        processed_corpus = map(lambda x: x + "\n", corpus)
        temp_file.writelines(processed_corpus)

    # Train model and delete temp file
    logger.debug("Instantiating FastText model `%s`", model_type)
    model = fasttext.train_unsupervised(corpus_path, model=model_type, epoch=10, thread=16)
    if os.path.exists(corpus_path):
        os.remove(corpus_path)

    def embedder(documents: List[str]) -> List[float]:
        """Generates an embedding using a FastText model"""
        return scale_vectors([model.get_sentence_vector(doc) for doc in documents])

    return embedder


def bert_embedder(model_name: str = "paraphrase-xlm-r-multilingual-v1",
                  device: str = 'cpu') -> Callable[[List[str]], List[float]]:
    """
    Given a corpus of texts and optionally a BERT model name, returns an embedding
    (representation of such texts) using the passed BERT pretrained model from the
    available list of Sentence BERT models.

    ref: https://www.sbert.net/
    """

    logger.debug("Instantiating BERT model `%s`", model_name)
    model = SentenceTransformer(model_name, device=device)

    def embedder(documents: List[str]) -> List[float]:
        """Generates an embedding using a FastText model"""
        return scale_vectors(model.encode(documents))

    return embedder


def reduce_dimensionality(embedder: Callable[[List[str]], List[float]],
                          dimensions: int = 50) -> Callable[[List[str]], List[float]]:
    """
    Given an embedder, returns a callable with reduced dimensionality using PCA
    """

    pca_model = PCA(n_components=dimensions)

    logger.debug("Reducing dimensionality of embedder %s to %s dimensions", embedder, dimensions)

    def new_embedder(documents: List[str]):
        """Calculates vectors using an external embedder and PCA to reduce dimensionality"""

        return scale_vectors(pca_model.fit_transform(embedder(documents)))

    return new_embedder


def get_embedder_functions(corpus: List[str]) -> Dict[str, Callable[[List[str]], List[float]]]:
    """
    Returns a list of the available embedders.
    #! If updated, update next function too
    """

    embedders = {
        # 'Bag of Words': bow_embedder(corpus),
        'FastText (CBOW)': fasttext_embedder(corpus, model_type="cbow"),
        'FastText (Skipgram)': fasttext_embedder(corpus, model_type="skipgram"),
        'Doc2Vec': doc2vec_embedder(corpus),
        'GPT2 Small Spanish': bert_embedder(model_name="datificate/gpt2-small-spanish"),
        'BERT: TinyBERT-spanish-uncased-finetuned-ner':
            bert_embedder(model_name='mrm8488/TinyBERT-spanish-uncased-finetuned-ner'),
        'BERT: paraphrase-xlm-r-multilingual-v1': bert_embedder(model_name='paraphrase-xlm-r-multilingual-v1'),
        'BERT: distiluse-base-multilingual-cased-v2': bert_embedder(model_name='distiluse-base-multilingual-cased-v2'),
    }

    reduced_embedders = {}
    for name, embedder in embedders.items():
        reduced_embedders[f"{name} (50-d)"] = reduce_dimensionality(embedder)

    return {**embedders, **reduced_embedders}


def get_embedder_function_names() -> List[str]:
    """
    Returns a list of the available embedder names
    """

    embedder_names = [
        'FastText (CBOW)',
        'FastText (Skipgram)',
        'Doc2Vec',
        'GPT2 Small Spanish',
        'BERT: TinyBERT-spanish-uncased-finetuned-ner',
        'BERT: paraphrase-xlm-r-multilingual-v1',
        'BERT: distiluse-base-multilingual-cased-v2',
    ]

    reduced_embedder_names = [f"{name} (50-d)" for name in embedder_names]

    return [*embedder_names, *reduced_embedder_names]


def get_embedder_function_short_names() -> Dict[str, str]:
    """
    Returns a mapping of short names for the available embedders names
    """

    embedder_names = {
        'FastText (CBOW)': 'FastText (CBOW)',
        'FastText (Skipgram)': 'FastText (SG)',
        'Doc2Vec': 'Doc2Vec',
        'GPT2 Small Spanish': 'GPT2 Small Sp.',
        'BERT: TinyBERT-spanish-uncased-finetuned-ner': 'BERT: TinyBERT',
        'BERT: paraphrase-xlm-r-multilingual-v1': 'BERT: paraphrase',
        'BERT: distiluse-base-multilingual-cased-v2': 'BERT: distiluse',
    }

    reduced_embedder_names = {
        f"{key} (50-d)": f"{value} (50-d)"
        for key, value in embedder_names.items()
    }

    return {**embedder_names, **reduced_embedder_names}


def main() -> None:
    """
    Run a small test program for embedders
    """

    test_sentences = [
        "yo soy tu padre",
        "este es mi amigo",
        "yo soy tu madre",
        "ese es mi padre",
        "ese es mi enemigo",
        "yo soy tu amigo fiel en las buenas y también en las malas que podamos pasar juntos o por separado",
        "esta es una frase",
        "esta es otra frase",
        "estoy intentando hacer espacio",
        "porque sino no puedo probar adecuadamente",
        "asi que sigo escribiendo frases",
        "un dos tres cuatro cinco",
    ]

    for name, embedder in get_embedder_functions(test_sentences).items():
        print(f"*** {name} ***")
        result_vectors = embedder(test_sentences)

        for sentence, vector in zip(test_sentences, result_vectors):
            print(f"-- Sentence `{ sentence }`, vector length `{ len(vector) }`")


if __name__ == '__main__':
    main()
