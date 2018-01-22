# #governmentshutdown2018 #trumpshutdown

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


nlp = spacy.load('en')
STOPLIST = set(list(ENGLISH_STOP_WORDS) + ["n't", "'s", "'m", "ca", "'", "'re",
                                           "pron"])
PUNCT_DICT = {ord(punc): None for punc in punctuation if punc not in ['_', '*']}
# ------------- set auth and initialize api -------------------------------
consumer_key = os.environ.get('twitter_consumer_key')
consumer_secret = os.environ.get('twitter_consumer_secret')
access_token = os.environ.get('twitter_access_token')
access_token_secret = os.environ.get('twitter_token_secret')
# -------------------------------------------------------------------------

class HashtagSentiment():
    
    def __init__(self, query, lang='en', show_plot=True):
        self.query = query
        self.lang = lang
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)
        self.tweets = []
        self.search_twitter()
        self.show_plot = show_plot
                
        
    def clean_text(self, tweet):
        '''
        remove links, usernames, and newlines from tweet
        (@\w+)               => removes usernames
        (#\w+)               => removes hashtags
        (\w+:\/\/\S+)        => removes links
        '''
        return re.sub('(@\w+)|(#\w+)|(\w+:\/\/\S+)|(\n+)', '', tweet)


    def get_sentiment(self):
        df = pd.DataFrame(self.tweets, columns=['text', 'date'])
 
        df['text'] = df['text'].apply(self.clean_text)
        sid = SentimentIntensityAnalyzer()
        df['sentiment'] = df.text.apply(lambda x: sid.polarity_scores(x)['compound'])
        df.drop('text', axis=1, inplace=True) 
        df.loc[df.sentiment >  0.1,'sentiment_type'] = 'pos'
        mask = (df.sentiment >= -0.1) & (df.sentiment <= 0.1)
        df.loc[mask, 'sentiment_type'] = 'neu'
        df.loc[df.sentiment <  -0.1,'sentiment_type'] = 'neg'
        df.date = pd.to_datetime(df.date)
        df['day'] = df.date.dt.day        
        # df = df.sort_values('day')
        day_gb = df.groupby('day')
        for day in day_gb.groups:
            day_df = df[df.day == day]
            print(day)
            print(day_df.sentiment.mean())
        
    
    def get_tweets(self, date):
        results = self.api.search(q=self.query, lang=self.lang, count=100,
                                  tweet_mode='extended', until=date)
        return results
        
        
        
    def search_twitter(self):
        for date in ['2018-01-18', '2018-01-19', '2018-01-20', '2018-01-21']:
            results = self.get_tweets(date)
            for tweet in results:
                tweet = tweet._json
                self.tweets.append((tweet['full_text'], tweet['created_at']))
            print(len(self.tweets))
            print('Sleeping...')
            time.sleep(2)
        
        self.get_sentiment()
    
            

if __name__ == '__main__':
    test = HashtagSentiment('%23trumpshutdown')





