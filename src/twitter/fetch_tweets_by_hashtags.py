"""
Small test script to fetch a couple of tweets using tweepy
around a given hashtag.

Usage:  python.py fetch_tweets_by_hashtag.py <hashtag> [<amount>]
"""

import tweepy
import os
import sys

#! IMPORTANT: Set the following environment vars
TW_CONSUMER_KEY = os.environ.get('TW_CONSUMER_KEY')
TW_CONSUMER_SECRET = os.environ.get('TW_CONSUMER_SECRET')
TW_ACCESS_TOKEN = os.environ.get('TW_ACCESS_TOKEN')
TW_ACCESS_SECRET = os.environ.get('TW_ACCESS_SECRET')

#? Sets the search within a 10km-radius in Caracas
LOCATION_GEOCODE = "10.4880,-66.8791,10km"

# Setting the keys on tweepy
auth = tweepy.OAuthHandler(TW_CONSUMER_KEY, TW_CONSUMER_SECRET)
auth.set_access_token(TW_ACCESS_TOKEN, TW_ACCESS_SECRET)

# Initializing the API handler
api = tweepy.API(auth)

def search_by_hashtag(hashtag, amount=50):
    """
    Returns a list of recently published tweets
    for a given hashtag near Caracas, Venezuela.
    """

    cursor = tweepy.Cursor(
        api.search,
        q=hashtag,
        geocode=LOCATION_GEOCODE
    ).items(amount)

    return list(cursor)

def main():
    """
    Main program. Reads the hashtag from standard input, then
    prints a list with those hashtags.
    """

    # Extract standard input arguments
    args = sys.argv[1:]

    # Print help and exit if there are no arguments
    if not args:
        print("Usage:\tpython.py fetch_tweets_by_hashtags.py <hashtag> [<amount>]")
        return sys.exit(1)

    hashtag = args[0]

    # If provided, parse second argument as the amount of tweets to fetch
    amount = 50
    if len(args) > 1:
        try:
            amount = int(args[1])
        except ValueError:
            pass

    # Gets tweets using the API
    try:
        tweets = search_by_hashtag(hashtag, amount)
    except tweepy.error.TweepError as e:
        print("Twitter API usage raised an error. %s" % e)
        print("Are the environment variables properly set and valid?")
        sys.exit(1)

    # Print each tweet information
    print("Printing up to {0} recent tweets near Caracas, Venezuela (10km radius) with the {1} hashtag...".format(
        amount,
        hashtag
    ))

    for tweet in tweets:
        print("Tweet by @{0} on {1}\n{2}\n".format(
            tweet.user.screen_name,
            tweet.created_at,
            tweet.text
        ))

    print("Done!")

# Runs the main program
if __name__ == "__main__":
    main()
