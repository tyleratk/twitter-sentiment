#################################################################
#     v1.03: adds adding it to mongodb
#            --------------
#     v1.02: corrects hashtags      
#            --------------
#     v1.01: update try/except for some attributes
#            adds extended_tweet
#            ---------------
#      todo: clean up code
#   working: gets 16 features about a tweet and saves it to csv
#################################################################

import tweepy
from tweepy import Stream
from tweepy.streaming import StreamListener
import csv
import time
import numpy as np
import pymongo
import clean_tweets
import os


# ------------- set auth and initialize api -------------------------------
consumer_key = os.environ.get('twitter_consumer_key')
consumer_secret = os.environ.get('twitter_consumer_secret')
access_token = os.environ.get('twitter_access_token')
access_token_secret = os.environ.get('twitter_token_secret')

auth = tweepy.OAuthHandler(consumer_key, consumer_secret) # authentication object
auth.set_access_token(access_token, access_token_secret) # access token and secret
api = tweepy.API(auth) # API object while passing in auth information


# --------------- stream data ---------------------------------
class MyListener(StreamListener):
    
    def on_status(self, data):
        if data.lang == 'en':
            tweet = data._json
            created_at = tweet['created_at']
            try:
                text = tweet['extended_tweet']['full_text']
            except:
                text = tweet['text']
            tweet_data = [created_at, text]
                          
            if save_csv:
                with open('../data/time_series_tweets.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(tweet_data)
                time.sleep(5)
            else:
                names = ['created_at', 'hash_tags', 'coordinates',
                         'coordinates_type', 'lang', 'country', 'tweet_location', 'text',
                         'user_created', 'default_profile_image', 'user_likes',
                         'user_followers', 'user_following', 'user_screen_name',
                         'user_num_tweets', 'user_location']
                tweet_json = dict(zip(names, tweet_data))
                tweet_json['text'] = clean_tweets.clean_text(tweet_json['text'])
                client = pymongo.MongoClient()
                db = client['tweet_data']
                table = db['tweets']
                table.insert_one(tweet_json)

            
            
    def on_error(self, status):
        print(status)
        return True



if __name__ == '__main__':
    save_csv=False
    
    twitter_stream = Stream(auth, MyListener())
    twitter_stream.filter(track='%23trumpshutdown', async=True)

    time.sleep(43200)
    twitter_stream.disconnect()




