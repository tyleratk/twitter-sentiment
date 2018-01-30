import tweepy
from tweepy import Stream
from tweepy.streaming import StreamListener
import csv
import time
import numpy as np
import pymongo
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

# ------------- set auth and initialize api -------------------------------
consumer_key = os.environ.get('twitter_consumer_key')
consumer_secret = os.environ.get('twitter_consumer_secret')
access_token = os.environ.get('twitter_access_token')
access_token_secret = os.environ.get('twitter_token_secret')

auth = tweepy.OAuthHandler(consumer_key, consumer_secret) # authentication object
auth.set_access_token(access_token, access_token_secret) # access token and secret
api = tweepy.API(auth) # API object while passing in auth information


def clean_text(tweet):
    '''
    remove links, usernames, and newlines from tweet
    (@\w+)               => removes usernames
    (#\w+)               => removes hashtags
    (\w+:\/\/\S+)        => removes links
    '''
    return re.sub('(@\w+)|(#\w+)|(\w+:\/\/\S+)|(\n+)', '', tweet)


def get_sentiment(text):
    text = clean_text(text)
    sid = SentimentIntensityAnalyzer()
    sent = sid.polarity_scores(text)['compound']
    return sent
    

# --------------- stream data ---------------------------------
class MyListener(StreamListener):
    
    def on_status(self, data):
        # geo, lang, place, text, user
        if data.lang == 'en':
            tweet = data._json
            created_at = tweet['created_at']
            try:
                coordinates = tweet['geo']['coordinates']
            except:
                coordinates = ''
            try:
                tweet_location = tweet['place']['full_name']
            except:
                tweet_location = ''
            try:
                text = tweet['extended_tweet']['full_text']
            except:
                text = tweet['text']
            try:
                user_location = tweet['user']['location']
            except:
                user_location = ''

            sentiment = get_sentiment(text)

            tweet_data = [created_at, coordinates, tweet_location, text,
                          user_location, sentiment]
                          

            if csv:
                with open('../data/super_bowl.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(tweet_data)
            
            
    def on_error(self, status):
        print(status)
        return True



if __name__ == '__main__':
    twitter_stream = Stream(auth, MyListener())
    twitter_stream.filter(track=['new england','patriots','pats','brady','super bowl'] ,locations=[-125,25,-65,48], async=True)

    time.sleep(1800)
    twitter_stream.filter(track=['philadelphia','eagles','super bowl'] ,locations=[-125,25,-65,48], async=True)
    time.sleep(1800)


    twitter_stream.disconnect()



