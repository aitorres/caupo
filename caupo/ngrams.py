"""
Auxiliary module for handlings, processing and obtaining bigrams
and potentially other n-grams
"""


from typing import List

from nltk import FreqDist
from nltk.util import ngrams
from nltk.tokenize import word_tokenize


def get_top_ngrams(phrases: List[str], top_n_amount: int = 3, ngram_size: int = 2) -> str:
    """Returns a string containing the `n` top bigrams of a set of phrases"""

    tokenized_phrases = [word_tokenize(phrase, language="spanish") for phrase in phrases]
    tokenized_phrases = [token for token in tokenized_phrases if len(token) > 2]
    ngrams_list = [ngrams(tkn_phrase, ngram_size) for tkn_phrase in tokenized_phrases]
    ngrams_flat_list = [ngram for ngram_list in ngrams_list for ngram in ngram_list]
    ngrams_distribution = FreqDist(ngrams_flat_list)
    ngrams_by_frequency = sorted(ngrams_distribution.items(), key=lambda item: (item[1], item[0]), reverse=True)
    top_n_ngrams = [" ".join(ngram).capitalize() for ngram, _ in ngrams_by_frequency[:top_n_amount]]
    return top_n_ngrams


def main() -> None:
    """Run a small test script"""

    phrases = [
        "yo soy una persona feliz",
        "la persona feliz no es una buena persona",
        "buenísimo, me caes muy bien",
        "eso no es así pana",
        "buena persona yo no soy",
        "yo sonoro sí soy",
        "qué persona yo puedo llegar a ser",
        "quisiera ser yo alguna persona feliz",
        "feliz como persona que yo soy",
    ]
    print(get_top_ngrams(phrases))


if __name__ == "__main__":
    main()
