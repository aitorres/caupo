"""
Small test script to fetch a couple of tweets using tweepy
around a given search term.

Usage:  python.py fetch_tweets_by_search_term.py <mode> <search_term>
"""

import tweepy
import os
import sys
import json
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

mongo_client = MongoClient('mongodb://127.0.0.1')
db = mongo_client.tweets

#! IMPORTANT: Set the following environment vars
TW_CONSUMER_KEY = os.environ.get('TW_CONSUMER_KEY')
TW_CONSUMER_SECRET = os.environ.get('TW_CONSUMER_SECRET')
TW_ACCESS_TOKEN = os.environ.get('TW_ACCESS_TOKEN')
TW_ACCESS_SECRET = os.environ.get('TW_ACCESS_SECRET')

#? Sets the search within a certain km radius in Caracas
KM_DISTANCE = 10
LOCATION_GEOCODE = f"10.4880,-66.8791,{KM_DISTANCE}km"

# Setting the keys on tweepy
auth = tweepy.OAuthHandler(TW_CONSUMER_KEY, TW_CONSUMER_SECRET)
auth.set_access_token(TW_ACCESS_TOKEN, TW_ACCESS_SECRET)

# Initializing the API handler
api = tweepy.API(auth, wait_on_rate_limit_notify=True, wait_on_rate_limit=True,
                 retry_count=3, retry_delay=5, retry_errors=set([401, 404, 500, 503]))

def get_search_cursor(search_term):
    """
    Returns an iterator of recently published tweets
    for a given search term near Caracas, Venezuela.
    """

    return tweepy.Cursor(
        api.search,
        q=f"{search_term} -filter:retweets lang:es",
        geocode=LOCATION_GEOCODE,
        tweet_mode="extended"
    ).items()

def main():
    """
    Main program. Reads the search term from standard input, then
    prints a list with those terms.
    """

    # Extract standard input arguments
    args = sys.argv[1:]

    # Print help and exit if there are no arguments
    if not args:
        print("Usage:\tpython.py fetch_tweets_by_search_term.py <mode> <search_term>")
        print("Modes: print|store")
        return sys.exit(1)

    mode = args[0].lower()
    search_term = args[1]

    if mode not in ["print", "store"]:
        print("Unknown mode")
        sys.exit(1)

    # Gets tweets using the API
    try:
        tweets = get_search_cursor(search_term)
    except tweepy.error.TweepError as e:
        print("Twitter API usage raised an error. %s" % e)
        print("Are the environment variables properly set and valid?")
        sys.exit(1)

    if mode == "print":
        # Print each tweet information
        print(f"Recent tweets near Caracas, Venezuela ({KM_DISTANCE}km radius) for `{search_term}` term...")

        for tweet in tweets:
            print("Tweet by @{0} on {1}\n{2}\n".format(
                tweet.user.screen_name,
                tweet.created_at,
                tweet.full_text
            ))

        print("Done!")
    elif mode == "store":
        for tweet in tweets:
            tweet_json = tweet._json
            tweet_json['created_at'] = str(tweet.created_at)
            tweet_json['_id'] = tweet.id
            tweet_json = json.dumps(tweet_json)
            try:
                db.tweets.insert(tweet_json)
                print(f"Successfully stored tweet {tweet.id}: {tweet.full_text}")
            except DuplicateKeyError:
                print(f"Skipping duplicate tweet {tweet.id}")

# Runs the main program
if __name__ == "__main__":
    main()
