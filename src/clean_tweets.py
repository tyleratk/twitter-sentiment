<<<<<<< HEAD
#################################################################
#     v1.0:
#     todo: fix clean_hashtags
#  working: clean tweet/hashtag 
# getting score of 97, but always predicts 0 => TRAIN ON SAME SIZED SAMPLES
#################################################################

=======
<<<<<<< HEAD
#################################################################
#     v1.0:
#     todo:
#  working: clean tweet/hashtag 
#################################################################

=======
>>>>>>> master
>>>>>>> df3963c6953c754fba9c03a23fb66f46690f8b7d
# open csv
# grab only the rows i am interested in
# remove links and usernames from tweet
#     make sure emoji's stay
# train model on emoji's
# classify tweets as happy or sad
<<<<<<< HEAD

import pandas as pd
import re
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


class TwitterClassifier():
    
    def __init__(self, classifier):
        self.pipeline = self.get_pipeline(classifier)
        
    def get_pipeline(self, classifier):
        if classifier.lower() == 'multinomial':
            vec = TfidfVectorizer(stop_words='english')
            classifier = MultinomialNB()
            return Pipeline(vec, classifier)
    
    def fit(X, y):
        pass
        

def clean_text(tweet):
=======
<<<<<<< HEAD

=======
>>>>>>> master
import pandas as pd
import re



def clean_tweet(tweet):
>>>>>>> df3963c6953c754fba9c03a23fb66f46690f8b7d
    '''
    remove links usernames and newlines from tweet
    (@\w+)               => removes usernames
    (#\w+)               => removes hashtags
    (\w+:\/\/\S+)        => removes links
    '''
<<<<<<< HEAD

    return re.sub('(@\w+)|(#\w+)|(\w+:\/\/\S+)|(\n+)', '', tweet)


# def clean_hashtags(row):      # dont know what its doing
#     '''
#     strips everything but hashtags
#     '''
#     if row == []:
#         return None
# 
#     hashtags = re.sub('[\W\d]', ' ', row).split()
#     return ' '.join(word for word in hashtags if word not in['text', 'indices'])
# 
#     return re.sub('(@\w+)|(#\w+)|(\w+:\/\/\S+)|(\n+)', '', tweet)    
    
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
    # df.text = df.text.apply(remove_emoji)
    gb = df.groupby('sentiment')
    ones = gb.get_group(1).sample(8000)
    zeros = gb.get_group(0).sample(8000)
    neg_ones = gb.get_group(-1).sample(8000)
    comb = pd.concat([ones, zeros, neg_ones])
    
    X = comb.text.values
    y = comb.sentiment.values
    
    return X, y
=======
<<<<<<< HEAD
    return re.sub('(@\w+)|(#\w+)|(\w+:\/\/\S+)|(\n+)', '', tweet)
    

def get_hashtags(row):
    '''
    strips everything but hashtags
    '''
    if row == []:
        return None
    
    hashtags = re.sub('[\W\d]', ' ', row).split()
    return ' '.join(word for word in hashtags if word not in['text', 'indices'])
=======
    return re.sub('(@\w+)|(#\w+)|(\w+:\/\/\S+)|(\n+)', '', tweet)    
>>>>>>> master
>>>>>>> df3963c6953c754fba9c03a23fb66f46690f8b7d
    

if __name__ == '__main__':
    
<<<<<<< HEAD
    # model = TwitterClassifier('multinomial')

=======
<<<<<<< HEAD
>>>>>>> df3963c6953c754fba9c03a23fb66f46690f8b7d
    # names = ['created_at', 'hash_tags', 'coordinates',
    #          'coordinates_type', 'lang', 'country', 'tweet_location', 'text',
    #          'user_created', 'default_profile_image', 'user_likes',
    #          'user_followers', 'user_following', 'user_screen_name',
    #          'user_num_tweets', 'user_location']
    # 
<<<<<<< HEAD
    # print('Loading data...')
    # df = pd.read_csv('../data/tweets.csv', names=names)
    # 
    # 
    # print('Cleaning tweets...')
    # df['text'] = df['text'].apply(clean_text)
    # 
    # 
    # happy = [b'\xf0\x9f\x98\x81', b'\xf0\x9f\x98\x8d', b'\xf0\x9f\x98\x82',
    #          b'\xf0\x9f\x98\x8a', b'\xf0\x9f\x98\x98', b'\xf0\x9f\x98\x89',
    #          b'\xf0\x9f\x98\x83', b'\xf0\x9f\xa4\xa3', b'\xf0\x9f\x98\x82',
    #          b'\xf0\x9f\x98\x97', b'\xf0\x9f\x98\x99', b'\xf0\x9f\x98\x9a',
    #          b'\xf0\x9f\x98\x9d', b'\xf0\x9f\x98\x9b', b'\xf0\x9f\x95\xba',
    #          b'\xf0\x9f\x92\x83', b'\xf0\x9f\x8e\x8a', b'\xf0\x9f\x8e\x89',
    #          b'\xf0\x9f\x92\x96']
    # 
    # sad = [b'\xf0\x9f\x98\x9e', b'\xf0\x9f\x98\x94', b'\xf0\x9f\x98\x9f',
    #        b'\xf0\x9f\x98\x95', b'\xf0\x9f\x99\x81', b'\xf0\x9f\x98\xa3',
    #        b'\xf0\x9f\x98\x96', b'\xf0\x9f\x98\xab', b'\xf0\x9f\x98\xa9',
    #        b'\xf0\x9f\x98\xa2', b'\xf0\x9f\x98\xad', b'\xf0\x9f\x98\xa4',
    #        b'\xf0\x9f\x98\xa0', b'\xf0\x9f\x98\xa1', b'\xf0\x9f\x98\xa8',
    #        b'\xf0\x9f\x98\xb0', b'\xf0\x9f\x98\xa5', b'\xf0\x9f\x98\x93',
    #        b'\xf0\x9f\xa4\xa2', b'\xf0\x9f\x98\xb7', b'\xf0\x9f\xa4\xa7',
    #        b'\xf0\x9f\xa4\x92', b'\xf0\x9f\x91\x8e', b'\xf0\x9f\x92\x94']
    # 
    # print('Getting sentiment...')
    # df['sentiment'] = df.apply(get_sentiment, axis=1)
    
    with open('tweets.pkl', 'rb') as infile:
        df = pickle.load(infile)
    
    vec = TfidfVectorizer()
    nb = MultinomialNB()
    
    # pipe = Pipeline([('TfidfVectorizer',vec)], [('MultinomialNB', classifier)])
    # pipe = make_pipeline(vec, classifier)


    
    # .sample(n)
    
    # y = df.sentiment.values
    X, y = get_x_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # print('Fitting model')
    # model = pipe.fit(X_train, y_train)
    # 
    # # y_pred = model.predict(X_test)
    # print('Score: {:.3f}'.format(model.score(X_test, y_test)))
    
    
    
    
=======
    # df = pd.read_csv('../data/tweets.csv', names=names)
    # 
    # tweets = df['text']
=======
    names = ['created_at', 'hash_tags', 'coordinates',
             'coordinates_type', 'lang', 'country', 'tweet_location', 'text',
             'user_created', 'default_profile_image', 'user_likes',
             'user_followers', 'user_following', 'user_screen_name',
             'user_num_tweets', 'user_location']

    df = pd.read_csv('../data/tweets.csv', names=names)

    tweets = df['text']
>>>>>>> master


>>>>>>> df3963c6953c754fba9c03a23fb66f46690f8b7d





