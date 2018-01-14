#################################################################
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


# ------------- set auth and initialize api -------------------------------
consumer_key = 'vv9NhyyyMTePWDBbNAkk7xly9'
consumer_secret = 'jRt3H4kIl2PAZQyV1m9PpcW96z1ncIVp68r2Dd1p94SyeWrm2t'
access_token = '948768906344312832-8DJ8eIZn01oNPIGgYBSEUeTu9azUGi0'
access_token_secret = 'uJ90fHHgYiyiWWDVwLkEbxQTnxzAXTmoHoJd8uHNA3tf9'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret) # authentication object
auth.set_access_token(access_token, access_token_secret) # access token and secret
api = tweepy.API(auth) # API object while passing in auth information


# --------------- stream data ---------------------------------
class MyListener(StreamListener):

    def on_status(self, data):
        # geo, lang, place, text, user
        if data.lang == 'en':#and data.geo != '':
            tweet = data._json
            created_at = tweet['created_at']
            hash_tags = tweet['entities']['hashtags']
            if hash_tags == []:
                hash_tags = None
            else:
                tags = []
                for tag in hash_tags:
                    tags.append(tag['text'])
                hash_tags = tags
            try:
                coordinates = tweet['geo']['coordinates']
            except:
                coordinates = ''
            try:
                coordinates_type = tweet['geo']['type']
            except:
                coordinates_type = ''
            lang = tweet['lang']
            try:
                country = tweet['place']['country']
            except:
                country = ''
            try:
                tweet_location = tweet['place']['full_name']
            except:
                tweet_location = ''
            try:
                text = tweet['extended_tweet']['full_text']
            except:
                text = tweet['text']
            user_created = tweet['user']['created_at']
            default_profile_image = tweet['user']['default_profile_image']
            user_likes = tweet['user']['favourites_count']
            user_followers = tweet['user']['followers_count']
            user_following = tweet['user']['friends_count']
            user_screen_name = tweet['user']['screen_name']
            user_num_tweets = tweet['user']['statuses_count']
            try:
                user_location = tweet['user']['location']
            except:
                user_location = ''

            tweet_data = [created_at, hash_tags, coordinates,
                          coordinates_type, lang, country, tweet_location, text,
                          user_created, default_profile_image, user_likes, 
                          user_followers, user_following, user_screen_name, 
                          user_num_tweets, user_location]
            
            with open('../data/tweets.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(tweet_data)
            
            
    def on_error(self, status):
        print(status)
        return True

twitter_stream = Stream(auth, MyListener())
# twitter_stream.sample(async=True)
twitter_stream.filter(locations=[-125,25,-65,48], async=True)

time.sleep(7200)
twitter_stream.disconnect()



