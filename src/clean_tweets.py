# open csv
# grab only the rows i am interested in
# remove links and usernames from tweet
#     make sure emoji's stay
# train model on emoji's
# classify tweets as happy or sad
import pandas as pd
import re



def clean_tweet(tweet):
    '''
    remove links usernames and newlines from tweet
    (@\w+)               => removes usernames
    (#\w+)               => removes hashtags
    (\w+:\/\/\S+)        => removes links
    '''
    return re.sub('(@\w+)|(#\w+)|(\w+:\/\/\S+)|(\n+)', '', tweet)    
    

if __name__ == '__main__':
    
    names = ['created_at', 'hash_tags', 'coordinates',
             'coordinates_type', 'lang', 'country', 'tweet_location', 'text',
             'user_created', 'default_profile_image', 'user_likes',
             'user_followers', 'user_following', 'user_screen_name',
             'user_num_tweets', 'user_location']

    df = pd.read_csv('../data/tweets.csv', names=names)

    tweets = df['text']







