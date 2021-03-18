"""
Auxiliary module that contains several functions useful for
common preprocessing tasks for the project
"""

from functools import partial
from itertools import filterfalse
from typing import List, Set

import emoji
import spacy
from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Install nltk data, if needed
nltk_download('stopwords')
nltk_download('punkt')

nlp = spacy.load('es_core_news_md')
stemmer = SnowballStemmer('spanish')


def lemmatizer(text: str) -> str:
    """
    Given a phrase, returns the same phrase lemmatized
    """

    doc = nlp(text)
    return ' '.join([word.lemma_ for word in doc])


def stem(text: str) -> str:
    """
    Given a text, returns the same text stemmed
    """

    return " ".join([stemmer.stem(i) for i in word_tokenize(text)])


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


def map_strange_characters(phrase):
    """Removes all accents (áéíóú) and linebreaks from a lowercase phrase"""

    accents_map = {
        'á': 'a',
        'é': 'e',
        'í': 'i',
        'ó': 'o',
        'ú': 'u',
        '\n': ' ',
    }

    return "".join(map(lambda x: x if x not in accents_map else accents_map[x], phrase))


def remove_urls_mentions_hashtags(phrase):
    """Removes all URLs, mentions and hashtags from a lowercase phrase"""

    def should_remove(token):
        """Determines whether a token should be removed"""

        if token.startswith("http"):
            return True

        if token.startswith("@"):
            return True

        if token.startswith("#"):
            return True

        return False


def preprocess_corpus(corpus: List[str], lemmatize: bool = True) -> List[str]:
    """
    Given a corpus of phrases / text, applies a series of functions
    that will preprocess the text and return a list of each preprocessed
    string (in the same order)
    """

    # Lowering case
    lowercase_corpus = map(lambda x: x.lower(), corpus)

    # Removing unnecessary elements (fit as needed)
    no_strange_tokens_corpus = map(remove_urls_mentions_hashtags, lowercase_corpus)
    character_mapped_corpus = map(map_strange_characters, no_strange_tokens_corpus)
    no_emoji_corpus = map(remove_emoji, character_mapped_corpus)

    # Temporarily corpus for further preprocessing
    splitted_corpus = map(word_tokenize, no_emoji_corpus)

    # Keep only alphanumeric strings
    alphanumeric_corpus = map(partial(filter, lambda x: x.isalpha()), splitted_corpus)

    # Remove stopwords
    # TODO: Maybe this should be done earlier
    spanish_stopwords = get_stopwords()
    clean_corpus = map(partial(filterfalse, lambda x: x in spanish_stopwords), alphanumeric_corpus)

    # Rejoin corpus
    corpus_list = map(list, clean_corpus)
    rejoined_corpus = map(" ".join, corpus_list)

    # Lemmatize and/or stem
    if lemmatize:
        final_corpus = list(map(lemmatizer, rejoined_corpus))
    else:
        final_corpus = list(map(stem, rejoined_corpus))

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
