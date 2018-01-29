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
            sentiment = get_sentiment(text)
            tweet_data = [text, sentiment]
            # import pdb; pdb.set_trace()
            filename = 'data/' + hash_tag + '.csv'
            if os.path.exists(filename):
                with open('data/' + hash_tag + '.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(tweet_data)
            else:
                with open('data/' + hash_tag + '.csv', 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(tweet_data)                
            
            
    def on_error(self, status):
        print(status)
        return True


def get_hashtag_info(auth, hashtag):
    global hash_tag
    hash_tag = hashtag
    
    twitter_stream = Stream(auth, MyListener())
    twitter_stream.filter(track=hashtag, async=True)
    print('Streaming')
    time.sleep(10)
    twitter_stream.disconnect()


if __name__ == '__main__':
    pass
    # ------------- set auth and initialize api -------------------------------
    # consumer_key = 'dSJjdS3K25Ff3wl8uqZFKFgIZ'
    # consumer_secret = '0z92aeUboBEXOFgmqSX1D6FYhfBv4LE5L14zz4OtcRYsIp5TbJ'
    # access_token = '948768906344312832-eCKxtrwvuZBfa92mKJ1TJTLvkfOmLec'
    # access_token_secret = 'ERVheG8crqQO0wQ4mmKhcVA5DLroFP80uoKFMTJKPgFZR'
    
    # twitter_stream = Stream(auth, MyListener())
    # twitter_stream.filter(track=['new england','patriots','pats','brady','super bowl'] ,locations=[-125,25,-65,48], async=True)
    # 
    # time.sleep(1800)
    # twitter_stream.filter(track=['philadelphia','eagles','super bowl'] ,locations=[-125,25,-65,48], async=True)
    # time.sleep(1800)


    # twitter_stream.disconnect()




