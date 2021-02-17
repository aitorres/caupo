import logging
import time

import emoji
import matplotlib.pyplot as plt
from pymongo import MongoClient

logger = logging.getLogger("caupo")

client = MongoClient('mongodb://127.0.0.1:27019')
db = client.caupo

class Timer:
    """Context handler to measure time of a function"""

    def __init__(self, action_name):
        self.action_name = action_name
        logger.info("Starting: %s", self.action_name)

        self.start_time = time.time()
        self.end_time = None
        self.duration = None

    def __enter__(self):
        return self.start_time

    def __exit__(self, type, value, traceback):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

        logger.info("Finishing: %s", self.action_name)
        logger.debug("Done in %.2f seconds.", self.duration)


def get_all_tweets():
    """
    Queries and returns a cursor with all tweets stored in the database.
    """

    return db.tweets.find()


def get_non_unique_tweets():
    """
    Queries and returns a cursor with all the tweets which content, taken as the `full_text`
    of the document structure, is repeated verbatim in another tweet.
    """

    return db.tweets.aggregate(
        [
            {
                "$group": {
                    "_id": "$full_text",
                    "count": {
                        "$sum": 1
                    }
                }
            },
            {
                "$match": {
                    "count": {
                        "$gt": 1
                    }
                }
            }
        ]
    )


def get_non_unique_content_from_tweets():
    """Returns a list with the text of non-unique tweets."""

    return [doc['_id'] for doc in get_non_unique_tweets()]


def get_text_from_all_tweets(exclude_uninteresting_usernames=True, exclude_uninteresting_text=True):
    """
    Queries and returns a cursor with the text from all texts (filtering out any
    other attributes)
    """

    if exclude_uninteresting_usernames:
        uninteresting_usernames = [
            "SismosVenezuela",
            "DolarBeta",
            "tiempo_caracas",
        ]
    else:
        uninteresting_usernames = []

    if exclude_uninteresting_text:
        uninteresting_text = get_non_unique_content_from_tweets()
    else:
        uninteresting_text = []

    return db.tweets.find(
        {
            "user.screen_name": { "$nin": uninteresting_usernames },
            "full_text": { "$nin": uninteresting_text },
        },
        {
            "full_text": 1
        }
    )


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


def plot_top_words(model, feature_names, n_top_words, title):
    """
    Given a Topic Modelling model from skleanr (LDA, NMF), generates a plot
    containing the top terms

    src: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
    """

    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
