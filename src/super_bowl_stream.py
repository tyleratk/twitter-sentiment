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
consumer_key = 'dSJjdS3K25Ff3wl8uqZFKFgIZ'
consumer_secret = '0z92aeUboBEXOFgmqSX1D6FYhfBv4LE5L14zz4OtcRYsIp5TbJ'
access_token = '948768906344312832-eCKxtrwvuZBfa92mKJ1TJTLvkfOmLec'
access_token_secret = 'ERVheG8crqQO0wQ4mmKhcVA5DLroFP80uoKFMTJKPgFZR'

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
            try:
                text = tweet['extended_tweet']['full_text']
            except:
                text = tweet['text']
            if any(t in topics for t in text.split()):
                pass
            else:
                return True
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
                user_location = tweet['user']['location']
            except:
                user_location = ''

            sentiment = get_sentiment(text)

            tweet_data = [text, created_at, coordinates, tweet_location,
                          user_location, sentiment]

            if csv:
                with open('../data/super_bowl.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(tweet_data)
            else:
                names = ['text','created_at', 'coordinates','tweet_location',
                         'user_location','sentiment']
                tweet_json = dict(zip(names, tweet_data))
                tweet_json['text'] = clean_text(tweet_json['text'])
                client = pymongo.MongoClient()
                db = client['tweet_data']
                table = db['sb_day']
                table.insert_one(tweet_json)

            time.sleep(.1)
    def on_error(self, status):
        print(status)
        return True


if __name__ == '__main__':
    csv = False
    topics = ['new england','patriots','pats','brady','super bowl', 
              'philadelphia','eagles','superbowl', 'sb52','sblii','superbowl52',
              'gronkowski','gronk']

    twitter_stream = Stream(auth, MyListener())
    twitter_stream.filter(track=topics, locations=[-125,25,-65,48], async=True)

    #time.sleep(3600)
    #twitter_stream.disconnect()




