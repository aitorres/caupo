"""
Auxiliary module for handlings, processing and obtaining bigrams
and potentially other n-grams
"""


from typing import List

from nltk import FreqDist
from nltk.util import ngrams
from nltk.tokenize import word_tokenize


def get_top_bigrams(phrases: List[str], top_n_amount: int = 2) -> str:
    """Returns a string containing the `n` top bigrams of a set of phrases"""

    tokenized_phrases = [word_tokenize(phrase) for phrase in phrases]
    bigrams_list = [ngrams(tkn_phrase, 2) for tkn_phrase in tokenized_phrases]
    bigrams = [bigram for bigram_list in bigrams_list for bigram in bigram_list]
    distribution = FreqDist(bigrams)
    bigrams_by_frequency = sorted(distribution.items(), key=lambda item: (item[1], item[0]), reverse=True)
    top_n_phrases = [" ".join(bigram).capitalize() for bigram, _ in bigrams_by_frequency[:top_n_amount]]
    return " - ".join(top_n_phrases)


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
    ]
    print(get_top_bigrams(phrases))


if __name__ == "__main__":
    main()
