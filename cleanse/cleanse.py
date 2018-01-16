import pandas as pd
import pymongo
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import pickle

def clean_text(tweet):
    '''
    remove links, usernames, and newlines from tweet
    (@\w+)               => removes usernames
    (#\w+)               => removes hashtags
    (\w+:\/\/\S+)        => removes links
    '''
    return re.sub('(@\w+)|(#\w+)|(\w+:\/\/\S+)|(\n+)', '', tweet)




if __name__ == '__main__':
    client = pymongo.MongoClient()
    collection = client.tweet_data
    table = collection.tweets
    df = pd.DataFrame(list(table.find()))
    print('Cleaning tweets...')
    df['text'] = df['text'].apply(clean_text)
    print('Getting sentiment...')
    sid = SentimentIntensityAnalyzer()
    df['sentiment'] = df.text.apply(lambda x: sid.polarity_scores(x)['compound'])   
    df.loc[df.sentiment >  0.5,'sentiment_type'] = 'pos'
    mask = (df.sentiment >= -0.5) & (df.sentiment <= 0.5)
    df.loc[mask, 'sentiment_type'] = 'neu'
    df.loc[df.sentiment <  -0.5,'sentiment_type'] = 'neg'
    
    with open('../data/clean_tweets.pkl', 'wb') as outfile:
        pickle.dump(df, outfile)
    print('Wrote clean_tweets.pkl')
    
