import pymongo

client = pymongo.MongoClient()
db = client.caupo


def get_all_tweets():
    """
    Queries and returns a cursor with all tweets stored in the database.
    """

    return db.tweets.find()

