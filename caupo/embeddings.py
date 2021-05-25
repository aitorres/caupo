"""
Auxiliary module to add implementations of several word embeddings
in order to test them for different tasks.
"""

import os
from functools import partial
from typing import Callable, Dict, List

import fasttext
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler


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

    tagged_documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(corpus)]
    model = Doc2Vec(tagged_documents, vector_size=150, window=5, min_count=3, workers=8)
    vectors = [model.infer_vector(doc.split()) for doc in corpus]

    return scale_vectors(vectors)


def bow_embedder(corpus: List[str]) -> List[float]:
    """
    Given a corpus of texts, returns an embedding (representation
    of such texts) using a Count Vectorizer for Bag of Words.

    ref: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    """

    model = CountVectorizer(min_df=3, max_df=0.9)
    vectors = model.fit_transform(corpus).toarray()

    return scale_vectors(vectors)


def fasttext_embedder(corpus: List[str], model_type: str = 'cbow') -> List[float]:
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
    model = fasttext.train_unsupervised(corpus_path, model=model_type)
    if os.path.exists(corpus_path):
        os.remove(corpus_path)

    # Get vectors
    vectors = [model.get_sentence_vector(doc) for doc in corpus]

    return scale_vectors(vectors)


def bert_embedder(corpus: List[str], model_name: str = "paraphrase-xlm-r-multilingual-v1",
                  device: str = 'cpu') -> List[float]:
    """
    Given a corpus of texts and optionally a BERT model name, returns an embedding
    (representation of such texts) using the passed BERT pretrained model from the
    available list of Sentence BERT models.

    ref: https://www.sbert.net/
    """

    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(corpus)

    return scale_vectors(vectors)


def reduce_dimensionality(embedder: Callable[[List[str]], List[float]],
                          dimensions: int = 50) -> Callable[[List[str]], List[float]]:
    """
    Given an embedder, returns a callable with reduced dimensionality using PCA
    """

    pca_model = PCA(n_components=dimensions)

    def new_embedder(corpus: List[str]):
        """Calculates vectors using an external embedder and PCA to reduce dimensionality"""

        vectors = embedder(corpus)
        reduced_vectors = pca_model.fit_transform(vectors)

        return reduced_vectors

    return new_embedder


def get_embedder_functions() -> Dict[str, Callable[[List[str]], List[float]]]:
    """
    Returns a list of the available embedders.

    # TODO: Add GloVe if possible
    """

    embedders = {
        'Bag of Words': bow_embedder,
        'Doc2Vec': doc2vec_embedder,
        'FastText (CBOW)': partial(fasttext_embedder, model_type="cbow"),
        'FastText (Skipgram)': partial(fasttext_embedder, model_type="skipgram"),
        'GPT2 Small Spanish': partial(  # ref: https://huggingface.co/datificate/gpt2-small-spanish
            bert_embedder, model_name="datificate/gpt2-small-spanish"),
        'BERT: TinyBERT-spanish-uncased-finetuned-ner': partial(
            bert_embedder, model_name='mrm8488/TinyBERT-spanish-uncased-finetuned-ner'),
        'BERT: paraphrase-xlm-r-multilingual-v1': partial(
            bert_embedder, model_name='paraphrase-xlm-r-multilingual-v1'),
        'BERT: distiluse-base-multilingual-cased-v2': partial(
            bert_embedder, model_name='distiluse-base-multilingual-cased-v2'),
    }

    reduced_embedders = {}
    for name, embedder in embedders.items():
        reduced_embedders[f"{name} (50-dim)"] = reduce_dimensionality(embedder)

    return {**embedders, **reduced_embedders}


def get_optimal_eps_for_embedder(distance: str, name: str) -> float:
    """
    Given the name of a distance and an embedder, returns the optimal value for the `eps`
    parameter that should be used in DBSCAN according to previous, human made analysis.
    """

    OPTIMAL_EPS = {
        'euclidean': {
            'Bag of Words': 175,
            'Doc2Vec': 12,
            'FastText (CBOW)': 5,
            'FastText (Skipgram)': 7,
            'GPT2 Small Spanish': 21,
            'BERT: TinyBERT-spanish-uncased-finetuned-ner': 15,
            'BERT: paraphrase-xlm-r-multilingual-v1': 30,
            'BERT: distiluse-base-multilingual-cased-v2': 25,
        },
        'cosine': {
            'Bag of Words': 0.65,
            'Doc2Vec': 0.15,
            'FastText (CBOW)': 0.075,
            'FastText (Skipgram)': 0.35,
            'GPT2 Small Spanish': 0.55,
            'BERT: TinyBERT-spanish-uncased-finetuned-ner': 0.325,
            'BERT: paraphrase-xlm-r-multilingual-v1': 0.55,
            'BERT: distiluse-base-multilingual-cased-v2': 0.55,
        }
    }

    try:
        return OPTIMAL_EPS[distance][name]
    except (IndexError, KeyError):
        return 0.5


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

    for name, embedder in get_embedder_functions().items():
        print(f"*** {name} ***")
        result_vectors = embedder(test_sentences)

        for sentence, vector in zip(test_sentences, result_vectors):
            print(f"-- Sentence `{ sentence }`, vector length `{ len(vector) }`")


if __name__ == '__main__':
    main()
