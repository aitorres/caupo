"""
Auxiliary module that performs sentiment analysis on corpus
"""

from typing import List

from sentiment_analysis_spanish import sentiment_analysis

sentiment = sentiment_analysis.SentimentAnalysisSpanish()


def calculate_average_sentiment(corpus: List[str]) -> float:
    """Given a collection of documents, returns the average sentiment analysis value"""

    return sum(sentiment.sentiment(phrase) for phrase in corpus) / len(corpus)
