import pandas as pd


names = ['created_at', 'hash_tags', 'coordinates',
         'coordinates_type', 'lang', 'country', 'tweet_location', 'text',
         'user_created', 'default_profile_image', 'user_likes',
         'user_followers', 'user_following', 'user_screen_name',
         'user_num_tweets', 'user_location']
         
df = pd.read_csv('tweets.csv', names=names)

print('There are {} tweets in the csv'.format(df.shape[1]))
print('Null percentage:\n{}'.format(df.isnull().sum()/df.shape[1]))
