#################################################################
#     v1.0:
#     todo: add more models, add gridsearch, add scoring, add tokenizer
#  working: linear_svc, nltk classification
# linear_svc - nltk :89%
# linear_svc_tfidf - nltk :86%
#################################################################
import pymongo
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, SVR
import string
punctuations = string.punctuation
from spacy.en import English
# parser = English()
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords 
from sklearn.pipeline import make_pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
import seaborn as sns




class TwitterClassifier():
    def __init__(self, classifier):
        self.classifier = classifier
        self.pipeline = self.get_pipeline(classifier)
        
        
    def get_pipeline(self, classifier):
        print('Creating model...')
        if classifier == 'linear_svc':
            vectorizer = CountVectorizer()
            model = LinearSVC()
        if classifier == 'linear_svc_tfidf':
            vectorizer = TfidfVectorizer()
            model = LinearSVC()
        if classifier == 'rbf':
            vectorizer = CountVectorizer()
            model = SVR()
            
        return make_pipeline(vectorizer, model)
        
        
    def clean_text(self, tweet):
        '''
        remove links, usernames, and newlines from tweet
        (@\w+)               => removes usernames
        (#\w+)               => removes hashtags
        (\w+:\/\/\S+)        => removes links
        '''
        return re.sub('(@\w+)|(#\w+)|(\w+:\/\/\S+)|(\n+)', '', tweet)
        
        
    def get_sentiment(self, row):
        '''
        returns sentiment based off of emoji
        '''
        happy = [b'\xf0\x9f\x98\x81', b'\xf0\x9f\x98\x8d', b'\xf0\x9f\x98\x82',
                 b'\xf0\x9f\x98\x8a', b'\xf0\x9f\x98\x98', b'\xf0\x9f\x98\x89',
                 b'\xf0\x9f\x98\x83', b'\xf0\x9f\xa4\xa3', b'\xf0\x9f\x98\x82',
                 b'\xf0\x9f\x98\x97', b'\xf0\x9f\x98\x99', b'\xf0\x9f\x98\x9a',
                 b'\xf0\x9f\x98\x9d', b'\xf0\x9f\x98\x9b', b'\xf0\x9f\x95\xba',
                 b'\xf0\x9f\x92\x83', b'\xf0\x9f\x8e\x8a', b'\xf0\x9f\x8e\x89',
                 b'\xf0\x9f\x92\x96']
        
        sad = [b'\xf0\x9f\x98\x9e', b'\xf0\x9f\x98\x94', b'\xf0\x9f\x98\x9f',
               b'\xf0\x9f\x98\x95', b'\xf0\x9f\x99\x81', b'\xf0\x9f\x98\xa3',
               b'\xf0\x9f\x98\x96', b'\xf0\x9f\x98\xab', b'\xf0\x9f\x98\xa9',
               b'\xf0\x9f\x98\xa2', b'\xf0\x9f\x98\xad', b'\xf0\x9f\x98\xa4',
               b'\xf0\x9f\x98\xa0', b'\xf0\x9f\x98\xa1', b'\xf0\x9f\x98\xa8',
               b'\xf0\x9f\x98\xb0', b'\xf0\x9f\x98\xa5', b'\xf0\x9f\x98\x93',
               b'\xf0\x9f\xa4\xa2', b'\xf0\x9f\x98\xb7', b'\xf0\x9f\xa4\xa7',
               b'\xf0\x9f\xa4\x92', b'\xf0\x9f\x91\x8e', b'\xf0\x9f\x92\x94']
        tweet = row.text.split()
        try:
            # use np.random incase there are multiple emojis
            emoji = np.random.choice([word.encode('utf-8') for word in tweet if word.encode('utf-8') in happy + sad])
        except:
            return 0
        if emoji in happy:
            return 1
        elif emoji in sad:
            return -1
        
        
    def remove_emoji(self, tweet):
        '''
        remove emojies from tweets
        '''
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', tweet) # no emoji
            
            
    def get_balanced_classes(self, df):
        '''
        balance out classes before training model so there isn't class imbalance
        '''
        if self.source == 'mongo':
            gb = df.groupby('sentiment')
            pos = gb.get_group(1).sample(9000)
            neu = gb.get_group(0).sample(9000)
            neg = gb.get_group(-1).sample(9000)
            comb = pd.concat([pos, neu, neg])
            X = comb.text.values
            y = comb.sentiment.values

        elif self.source == 'nltk':
            gb = df.groupby('sentiment_type')
            pos = gb.get_group('pos').sample(18000)
            neu = gb.get_group('neu').sample(18000)
            neg = gb.get_group('neg').sample(18000)
            comb = pd.concat([pos, neu, neg])
            X = comb.text.values
            y = comb.sentiment_type.values

        return X, y
            
            
    def load_data(self, source):
        '''
        load data from mongo, clean tweets, and get sentiment
        '''
        if source == 'mongo':
            print('Loading data...')
            client = pymongo.MongoClient()
            collection = client.tweet_data
            table = collection.tweets
            df = pd.DataFrame(list(table.find()))
            print('Cleaning tweets...')
            df['text'] = df['text'].apply(self.clean_text)
            print('Getting sentiment...')
            df['sentiment'] = df.apply(self.get_sentiment, axis=1)
            df = df[['text', 'sentiment']]
            df['text'] = df['text'].apply(self.remove_emoji)
            tweets, labels = self.get_balanced_classes(df)
            # with open('../data/mongo_pkl.pkl', 'wb') as outfile:
            #     pickle.dump([tweets, labels], outfile)
            #     print('Wrote mongo_pkl')
            
        if source == 'nltk':
            print('Loading data...')
            client = pymongo.MongoClient()
            collection = client.tweet_data
            table = collection.tweets
            df = pd.DataFrame(list(table.find()))
            print('Cleaning tweets...')
            df['text'] = df['text'].apply(self.clean_text)
            print('Getting sentiment...')
            # df['text'] = df['text'].apply(self.remove_emoji)
            sid = SentimentIntensityAnalyzer()
            df['sentiment'] = df.text.apply(lambda x: sid.polarity_scores(x)['compound'])   
            df = df[['text', 'sentiment']]
            df.loc[df.sentiment >  0.5,'sentiment_type'] = 'pos'
            mask = (df.sentiment >= -0.5) & (df.sentiment <= 0.5)
            df.loc[mask, 'sentiment_type'] = 'neu'
            df.loc[df.sentiment <  -0.5,'sentiment_type'] = 'neg'
            # tweets, labels = self.get_balanced_classes(df)
            tweets, labels = df.text.values, df.sentiment_type.values

            with open('../data/nltk_pkl.pkl', 'wb') as outfile:
                pickle.dump([tweets, labels], outfile)
                print('Wrote nltk_pkl')

            
        if source == 'mongo_pkl':
            with open('../data/mongo_pkl.pkl', 'rb') as infile:
                tweets, labels = pickle.load(infile)


        if source == 'nltk_pkl':
            with open('../data/nltk_pkl.pkl', 'rb') as infile:
                tweets, labels = pickle.load(infile)
                
        return tweets, labels

        
    def train(self, source):
        '''
        get clean data and fit pipeline
        '''
        self.source = source
        
        tweets, labels = self.load_data(source)
        X_train, X_test, y_train, y_test = train_test_split(tweets, labels)
        self.pipeline.fit(X_train, y_train)
        print('Score: {:.2f}'.format(self.pipeline.score(X_test, y_test)))
        
        # self.pipeline.fit(tweets, labels)
        
        
    def predict(self, tweet):
        '''
        return sentiment of tweet
        '''
        return self.pipeline.predict(tweet)
    





if __name__ == '__main__':
    model = TwitterClassifier('linear_svc_tfidf')
    model.train('nltk') 
    
 

    
    
    



