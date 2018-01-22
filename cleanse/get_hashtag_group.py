import pandas as pd
import pickle
import re


def clean_hashtags(row):
    '''
    strips everything but hashtags
    '''
    if row == [] or row is None:
        return None
        
    hashtags = re.sub('[\W\d]', ' ', row).split()
    return ' '.join(word for word in hashtags if word not in['text', 'indices'])

    
    


if __name__ == '__main__':
    
    with open('../data/clean_tweets.pkl', 'rb') as infile:
        df = pickle.load(infile)
    print('Size before cleaning: {}'.format(df.shape[0]))
    df.hash_tags = df.hash_tags.apply(clean_hashtags)
    df.dropna(axis=0, inplace=True)
    print('Size after cleaning: {}'.format(df.shape[0]))
    df = df[df.hash_tags != '']
    print('Size after cleaning x2: {}'.format(df.shape[0]))
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    with open('../data/clean_tweets_hashtags.pkl', 'wb') as outfile:
        pickle.dump(df, outfile)
    print('Wrote pkl')
    

    
        
    
        
    