#################################################################
#     v1.0:
#     todo: fix clean_hashtags
#  working: clean tweet/hashtag
# getting score of 97, but always predicts 0 => TRAIN ON SAME SIZED SAMPLES
#################################################################


import pandas as pd
import re
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


def clean_tweet(tweet):
    '''
    remove links usernames and newlines from tweet
    (@\w+)               => removes usernames
    (#\w+)               => removes hashtags
    (\w+:\/\/\S+)        => removes links
    '''
    return re.sub('(@\w+)|(#\w+)|(\w+:\/\/\S+)|(\n+)', '', tweet)


def get_sentiment(row):
    tweet = row.text.split()
    try:
        emoji = np.random.choice([word.encode('utf-8') for word in tweet if word.encode('utf-8') in happy + sad])
    except:
        return 0
    if emoji in happy:
        return 1
    elif emoji in sad:
        return -1


def remove_emoji(tweet):

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tweet) # no emoji


def get_x_y():
    gb = df.groupby('sentiment')
    ones = gb.get_group(1).sample(8000)
    zeros = gb.get_group(0).sample(8000)
    neg_ones = gb.get_group(-1).sample(8000)
    comb = pd.concat([ones, zeros, neg_ones])

    X = comb.text.values
    y = comb.sentiment.values

    return X, y


if __name__ == '__main__':
    pass
