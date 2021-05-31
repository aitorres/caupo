import logging
import time
from typing import List

import bson.regex
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from caupo.database import get_db

mpl.use('Agg')

# Instantiate logger
logger = logging.getLogger("caupo")

db = get_db()


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


def get_city_modes():
    """Returns a dictionary with options for city modes to use in standarized measurements"""

    # ? INFO: Elements are of the form (key, value) === (title, parameter)
    CITY_MODES = {
        'Caracas': 'Caracas',
        'All Cities': None,
    }

    return CITY_MODES


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


def get_uninteresting_usernames():
    """Returns a list with the user handles that are not interesting for queries"""

    return [
        "SismosVenezuela",
        "DolarBeta",
        "tiempo_caracas",
    ]


def get_main_corpus() -> List[str]:
    """Returns the list of tweets (content as text) to consider as the main corpus"""

    return list(set(get_text_from_all_tweets(city="Caracas")))


def get_text_from_all_tweets(exclude_uninteresting_usernames=True, exclude_uninteresting_text=True,
                             city=None, dates=None):
    """
    Queries and returns a cursor with the text from all texts (filtering out any
    other attributes)
    """

    if exclude_uninteresting_usernames:
        uninteresting_usernames = get_uninteresting_usernames()
    else:
        uninteresting_usernames = []

    if exclude_uninteresting_text:
        uninteresting_text = get_non_unique_content_from_tweets()
    else:
        uninteresting_text = []

    query = {
        "user.screen_name": { "$nin": uninteresting_usernames },
        "full_text": { "$nin": uninteresting_text },
    }

    if dates is not None:
        query["created_at"] = {
            "$in": [bson.regex.Regex(f"^{d}") for d in dates]
        }

    if city is not None:
        query["city_tag"] = city

    tweets = db.tweets.find(
        query,
        { "full_text": 1 }
    )

    return [t["full_text"] for t in tweets]


def plot_clusters(vectors, filename, title, labels=None):
    """
    Given a numpy array with (2D-assumed) vectors, a filename, and a series of labels, stores
    a scatterplot and stores it as a PNG image in the received filename.

    src: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
    src: https://matplotlib.org/stable/tutorials/colors/colors.html
    """

    if len(vectors) == 0:
        logger.warning("Can't plot empty clusters!")
        return

    # Reducing dimensionality if needed
    dims = len(vectors[0])
    if dims > 2:
        logger.debug("Reducing dimensionality of vectors from `%s` to 2 before plotting", dims)
        pca_model = PCA(n_components=2)
        vectors = pca_model.fit_transform(vectors)

    # If no labels are passed, assume all belong to the same cluster (and are colored likewise)
    if labels is None:
        labels = [0] * len(vectors)

    if len(set(labels)) <= 11:
        color_palette = ["#00394B", "#005B6E", "#04668C", "#3C6CA7", "#726EB7",
                         "#A86BBA", "#DA66AC", "#FF6792", "#FF89AC", "#FFACBF"]
    elif len(set(labels) <= 31):
        color_palette = ["#012733", "#0A2C37", "#13303B", "#1B353F", "#243944",
                         "#2D3E48", "#36424C", "#3E4750", "#474B54", "#505058",
                         "#59555C", "#615961", "#6A5E65", "#736269", "#7C676D",
                         "#846B71", "#8D7075", "#967479", "#9F797D", "#A77D82",
                         "#B08286", "#B9878A", "#C28B8E", "#CA9092", "#D39496",
                         "#DC999A", "#E59D9F", "#EDA2A3", "#F6A6A7", "#FFABAB"]
    else:
        logger.warning("Won't plot more than 31 clusters!")
        return

    color_palette.append("#A6A6A6")  # gray would be used for a "-1" label)

    colors = [color_palette[i] for i in labels]
    plt.scatter(vectors[:, 0], vectors[:, 1], c=colors, s=2)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


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
