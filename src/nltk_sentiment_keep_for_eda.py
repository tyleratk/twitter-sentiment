import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import re


def clean_text(tweet):
    '''
    remove links usernames and newlines from tweet
    (@\w+)               => removes usernames
    (#\w+)               => removes hashtags
    (\w+:\/\/\S+)        => removes links
    '''

    return re.sub('(@\w+)|(#\w+)|(\w+:\/\/\S+)|(\n+)', '', tweet)
    
    
def remove_emoji(tweet):

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tweet) # no emoji
    

def get_x_y(df):
    df.loc[df.sentiment >  0,'sentiment_type'] = 'pos'
    df.loc[df.sentiment == 0,'sentiment_type'] = 'neu'
    df.loc[df.sentiment <  0,'sentiment_type'] = 'neg'

    gb = df.groupby('sentiment_type')
    pos = gb.get_group('pos').sample(18000)
    neu = gb.get_group('neu').sample(18000)
    neg = gb.get_group('neg').sample(18000)
    comb = pd.concat([pos, neu, neg])
    return comb



if __name__ == '__main__':
    names = ['created_at', 'hash_tags', 'coordinates',
             'coordinates_type', 'lang', 'country', 'tweet_location', 'text',
             'user_created', 'default_profile_image', 'user_likes',
             'user_followers', 'user_following', 'user_screen_name',
             'user_num_tweets', 'user_location']

    df = pd.read_csv('~/twitter/twitter/data/tweets.csv', names=names)
    
    df['text'] = df['text'].apply(clean_text)
    df['text'] = df['text'].apply(remove_emoji)
    
    sid = SentimentIntensityAnalyzer()
    df['sentiment'] = df.text.apply(lambda x: sid.polarity_scores(x)['compound'])   
    
    sns.violinplot(df.sentiment)
    equal_x_y = get_x_y(df)
    sns.violinplot(equal_x_y.sentiment_type)
     
    plt.show()   

    