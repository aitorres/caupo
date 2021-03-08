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
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

def doc2vec_embedder(corpus: List[str]) -> List[float]:
    """
    Given a corpus of texts, returns an embedding (representation
    of such texts) using a fine-tuned Doc2Vec embedder.

    ref: https://radimrehurek.com/gensim/models/doc2vec.html
    """

    tagged_documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(corpus)]
    model = Doc2Vec(tagged_documents, vector_size=100, window=5, min_count=3, workers=2)
    vectors = [model.infer_vector(doc.split()) for doc in corpus]

    return vectors


def bow_embedder(corpus: List[str]) -> List[float]:
    """
    Given a corpus of texts, returns an embedding (representation
    of such texts) using a Count Vectorizer for Bag of Words.

    ref: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    """

    model = CountVectorizer(min_df=3, max_df=0.9)
    vectors = model.fit_transform(corpus).toarray()

    return vectors


def fasttext_embedder(corpus: List[str], model_type: str = 'cbow') -> List[float]:
    """
    Given a corpus of texts and optionally a model type, returns an embedding
    (representation of such texts) using Fast Text as the embedder.

    TODO: Probably has a bug while reading input
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

    return vectors


def bert_embedder(corpus: List[str], model_name: str = "paraphrase-xlm-r-multilingual-v1",
                  device: str = 'cpu') -> List[float]:
    """
    Given a corpus of texts and optionally a BERT model name, returns an embedding
    (representation of such texts) using the passed BERT pretrained model from the
    available list of Sentence BERT models.

    ref: https://www.sbert.net/
    """

    model = SentenceTransformer(model_name , device=device)
    vectors = model.encode(corpus)

    return vectors


def reduce_dimensionality(corpus: List[str], embedder: Callable[[List[str]], List[float]], dimensions: int = 10):
    """
    Given an embedder and a list of (final) dimensions, returns another indirect embedder
    that always returns at most the requested dimensions using UMAP for dimensionality
    reduction (if needed).
    """

    original_vectors = embedder(corpus)
    vectors = UMAP(n_neighbors=10, n_components=dimensions, metric='cosine').fit_transform(original_vectors)

    return vectors


def get_embedder_functions() -> Dict[str, Callable[[List[str]], List[float]]]:
    """
    Returns a list of the available embedders.

    # TODO: Add GloVe if possible
    """

    regular_embedders = {
        'Bag of Words': bow_embedder,
        'Doc2Vec': doc2vec_embedder,
        'FastText (CBOW)': partial(fasttext_embedder, model_type="cbow"),
        'FastText (Skipgram)': partial(fasttext_embedder, model_type="skipgram"),
        'GPT2 Small Spanish': partial( # ref: https://huggingface.co/datificate/gpt2-small-spanish
            bert_embedder, model_name="datificate/gpt2-small-spanish"),
        'BERT: paraphrase-xlm-r-multilingual-v1': partial(
            bert_embedder, model_name='paraphrase-xlm-r-multilingual-v1'),
        'BERT: distiluse-base-multilingual-cased-v2': partial(
            bert_embedder, model_name='distiluse-base-multilingual-cased-v2'),
        'BERT: stsb-xlm-r-multilingual': partial(
            bert_embedder, model_name='stsb-xlm-r-multilingual'),
        'BERT: xlm-r-100langs-bert-base-nli-mean-tokens': partial(
            bert_embedder, model_name='xlm-r-100langs-bert-base-nli-mean-tokens'),
        'BERT: Geotrend/bert-base-es-cased': partial(
            bert_embedder, model_name='Geotrend/bert-base-es-cased'),
        'BERT: TinyBERT-spanish-uncased-finetuned-ner': partial(
            bert_embedder, model_name='mrm8488/TinyBERT-spanish-uncased-finetuned-ner'),
    }

    DIMENSIONS_TO_REDUCE = 10

    reduced_embedders = {}
    for name, embedder in regular_embedders.items():
        reduced_embedders[f"{name} (reduced, {DIMENSIONS_TO_REDUCE} dimensions)"] = partial(
            reduce_dimensionality, embedder=embedder, dimensions=DIMENSIONS_TO_REDUCE)

    #? INFO: Reduced embedders are taking way too much RAM, deactivated for now
    embedders = {**regular_embedders, **reduced_embedders}

    return regular_embedders


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
