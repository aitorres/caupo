"""
Test script for comparing or contrasting different runs of Doc2Vec
"""

import os
from typing import List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from caupo.embeddings import scale_vectors
from caupo.preprocessing import preprocess_v2
from caupo.tags import get_tags_by_frequency, fetch_tag_from_db
from caupo.utils import get_main_corpus, plot_clusters


def doc2vec_embedder(corpus: List[str], scale: bool = True, size: int = 100, window: int = 5) -> List[float]:
    """
    Given a corpus of texts, returns an embedding (representation
    of such texts) using a fine-tuned Doc2Vec embedder.

    ref: https://radimrehurek.com/gensim/models/doc2vec.html
    """

    print(f"Training Doc2Vec with: scale={scale}, size={size}, window={window}")
    tagged_documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(corpus)]
    model = Doc2Vec(tagged_documents, vector_size=size, window=window, min_count=3, workers=16)

    def embedder(documents: List[str]) -> List[float]:
        """Generates an embedding using a Doc2Vec"""
        return scale_vectors([model.infer_vector(doc.split()) for doc in documents], scale=scale)

    return embedder


def main() -> None:
    """Run script for tests"""

    windows = [3, 5, 7]
    sizes = [50, 100, 150, 200, 400]
    scales = [True, False]
    output_path = "outputs/exploration/doc2vecs"
    frequency = "daily"
    os.makedirs(output_path, exist_ok=True)

    print("Getting main corpus")
    corpus = get_main_corpus()
    print("Cleaning main corpus")
    cleaned_corpus = list(set(map(preprocess_v2, corpus)))

    tag_name, _ = get_tags_by_frequency(frequency)[0]
    print("Getting tweets of %s", tag_name)
    tag = fetch_tag_from_db(frequency, tag_name)
    tweets = tag["tweets"]

    print("Cleaning tweets")
    cleaned_tweets = list(set(map(preprocess_v2, tweets)))

    for window in windows:
        for size in sizes:
            for scale in scales:
                print(f"Now trying scale={scale}, size={size}, window={window}")
                model = doc2vec_embedder(cleaned_corpus, scale, size, window)
                vectors = model(cleaned_tweets)

                print("Plotting...")
                plot_clusters(vectors, f"{output_path}/window{window}-size{size}-scale{scale}.png",
                              f"scale={scale}, size={size}, window={window}")
    print("Done!")


if __name__ == "__main__":
    main()
