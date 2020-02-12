"""
Small test script to fetch a couple of tweets using tweepy
around a given search term.

Usage:  python.py fetch_tweets_by_search_term.py <search_term> [<amount>]
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

def search_by_term(search_term, amount=25):
    """
    Returns a list of recently published tweets
    for a given search term near Caracas, Venezuela.
    """

    cursor = tweepy.Cursor(
        api.search,
        q=search_term,
        geocode=LOCATION_GEOCODE
    ).items(amount)

    return list(cursor)

def main():
    """
    Main program. Reads the search term from standard input, then
    prints a list with those terms.
    """

    # Extract standard input arguments
    args = sys.argv[1:]

    # Print help and exit if there are no arguments
    if not args:
        print("Usage:\tpython.py fetch_tweets_by_search_term.py <search_term> [<amount>]")
        return sys.exit(1)

    search_term = args[0]

    # If provided, parse second argument as the amount of tweets to fetch
    amount = 25
    if len(args) > 1:
        try:
            amount = int(args[1])
        except ValueError:
            pass

    # Gets tweets using the API
    try:
        tweets = search_by_term(search_term, amount)
    except tweepy.error.TweepError as e:
        print("Twitter API usage raised an error. %s" % e)
        print("Are the environment variables properly set and valid?")
        sys.exit(1)

    # Print each tweet information
    print("Printing up to {0} recent tweets near Caracas, Venezuela (10km radius) with the {1} search term...".format(
        amount,
        search_term
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