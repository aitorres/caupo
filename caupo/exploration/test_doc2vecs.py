"""
Test script for comparing or contrasting different runs of Doc2Vec
"""

import logging
import os
from typing import List
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from caupo.embeddings import scale_vectors
from caupo.preprocessing import preprocess_v2
from caupo.tags import get_tags_by_frequency, fetch_tag_from_db
from caupo.utils import get_main_corpus, plot_clusters

logger = logging.getLogger("caupo")


def doc2vec_embedder(corpus: List[str], size: int = 100, window: int = 5) -> List[float]:
    """
    Given a corpus of texts, returns an embedding (representation
    of such texts) using a fine-tuned Doc2Vec embedder.

    ref: https://radimrehurek.com/gensim/models/doc2vec.html
    """

    logger.info(f"Training Doc2Vec with: size={size}, window={window}")
    tagged_documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(corpus)]
    model = Doc2Vec(tagged_documents, vector_size=size, window=window, min_count=3, workers=16)

    def embedder(documents: List[str]) -> List[float]:
        """Generates an embedding using a Doc2Vec"""
        return scale_vectors([model.infer_vector(doc.split()) for doc in documents])

    return embedder


def main() -> None:
    """Run script for tests"""

    windows = [3, 5, 7]
    sizes = [50, 100, 150]
    output_path = "outputs/exploration/doc2vecs"
    frequency = "daily"
    os.makedirs(output_path, exist_ok=True)

    logger.info("Getting main corpus")
    corpus = get_main_corpus()
    tag_name, _ = get_tags_by_frequency(frequency)[0]
    logger.info(f"Getting tweets of {tag_name}")

    for stem in [False, True]:
        logger.info(f"Using {stem} for stemming")

        logger.info("Cleaning main corpus")
        cleaned_corpus = list(set(map(lambda t: preprocess_v2(t, stem), corpus)))

        tag = fetch_tag_from_db(frequency, tag_name)
        tweets = tag["tweets"]

        logger.info("Cleaning tweets")
        cleaned_tweets = list(set(map(lambda t: preprocess_v2(t, stem), tweets)))

        for window in windows:
            for size in sizes:
                logger.info(f"Now trying size={size}, window={window}, stem={stem}")
                model = doc2vec_embedder(cleaned_corpus, size, window)
                vectors = model(cleaned_tweets)

                logger.info("Plotting...")
                plot_clusters(vectors, f"{output_path}/window{window}-size{size}-stem{stem}.png",
                              f"size={size}, window={window}, stem={stem}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
