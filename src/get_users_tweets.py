import pickle
import pandas as pd
from textacy.vsm import Vectorizer
import textacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
import spacy
import tweepy
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import os
import time
import networkx as nx

# ------------- set auth and initialize api -------------------------------
consumer_key = 'dSJjdS3K25Ff3wl8uqZFKFgIZ'
consumer_secret = '0z92aeUboBEXOFgmqSX1D6FYhfBv4LE5L14zz4OtcRYsIp5TbJ'
access_token = '948768906344312832-eCKxtrwvuZBfa92mKJ1TJTLvkfOmLec'
access_token_secret = 'ERVheG8crqQO0wQ4mmKhcVA5DLroFP80uoKFMTJKPgFZR'


# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth)

# --------------- print 20 tweets from user 'nytimes' -----------------------
name = "nytimes"
tweetCount = 20
results = api.user_timeline(id=name, count=tweetCount)
for tweet in results:
    tweet = tweet._json
    