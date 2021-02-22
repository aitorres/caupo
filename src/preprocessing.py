"""
Auxiliary module that contains several functions useful for
common preprocessing tasks for the project
"""

from functools import partial
from itertools import filterfalse
from typing import List, Set

import emoji
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Install nltk data, if needed
nltk_download('stopwords')
nltk_download('punkt')


def get_stopwords() -> Set[str]:
    """
    Returns a set with the stopwords to consider in the project
    """

    spanish_stopwords = set(stopwords.words('spanish'))

    # We manually add laughter
    laughter = {"ja" * i for i in range(1, 7)}

    return spanish_stopwords.union(laughter)


def remove_emoji(phrase):
    """Removes all emojis from a phrase"""

    return emoji.get_emoji_regexp().sub(r'', phrase)


def remove_accents(phrase):
    """Removes all accents (áéíóú) from a lowercase phrase"""

    accents_map = {
        'á': 'a',
        'é': 'e',
        'í': 'i',
        'ó': 'o',
        'ú': 'u',
    }

    return "".join(map(lambda x: x if x not in accents_map else accents_map[x], phrase))


def remove_linebreaks(phrase):
    """Removes all linebreaks from a phrase, replacing by whitespace"""

    return "".join(map(lambda x: " " if x == "\n" else x, phrase))


def remove_urls(phrase):
    """Removes all URLs from a lowercase phrase"""

    return " ".join([token for token in phrase.split() if not token.startswith("http")])


def remove_mentions(phrase):
    """Removes all Twitter mentions (@username) from a lowercase phrase"""

    return " ".join([token for token in phrase.split() if not token.startswith("@")])


def remove_hashtags(phrase):
    """Removes all Twitter hashtags (#username) from a lowercase phrase"""

    return " ".join([token for token in phrase.split() if not token.startswith("#")])


def preprocess_corpus(corpus: List[str]) -> List[str]:
    """
    Given a corpus of phrases / text, applies a series of functions
    that will preprocess the text and return a list of each preprocessed
    string (in the same order)
    """

    # Lowering case
    lowercase_corpus = map(lambda x: x.lower(), corpus)

    # Removing unnecessary elements (fit as needed)
    no_urls_corpus = map(remove_urls, lowercase_corpus)
    no_mentions_corpus = map(remove_mentions, no_urls_corpus)
    no_hashtags_corpus = map(remove_hashtags, no_mentions_corpus)
    unaccented_corpus = map(remove_accents, no_hashtags_corpus)
    no_emoji_corpus = map(remove_emoji, unaccented_corpus)
    no_linebreaks_corpus = map(remove_linebreaks, no_emoji_corpus)

    # Temporarily corpus for further preprocessing
    splitted_corpus = map(word_tokenize, no_linebreaks_corpus)

    # Keep only alphanumeric strings # TODO: think better about this
    alphanumeric_corpus = map(partial(filter, lambda x: x.isalpha()), splitted_corpus)

    # Remove stopwords
    spanish_stopwords = get_stopwords()
    clean_corpus = map(partial(filterfalse, lambda x: x in spanish_stopwords), alphanumeric_corpus)

    # Rejoin corpus
    corpus_list = list(map(list, clean_corpus))
    final_corpus = list(map(" ".join, corpus_list))

    return final_corpus


def main() -> None:
    """
    Run a small test
    """

    test_corpus = [
        "Esta es una frase! Qué loco todo! @andres #tema",
        """😋
        BASADO
        😋"""
    ]
    preprocessed_corpus = preprocess_corpus(test_corpus)

    for phrase in preprocessed_corpus:
        print(f"(*) {phrase}")


if __name__ == '__main__':
    main()
