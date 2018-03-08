import tweepy


# ------------- set auth and initialize api -------------------------------
consumer_key = os.environ.get('twitter_consumer_key')
consumer_secret = os.environ.get('twitter_consumer_secret')
access_token = os.environ.get('twitter_access_token')
access_token_secret = os.environ.get('twitter_token_secret')

auth = tweepy.OAuthHandler(consumer_key, consumer_secret) # authentication object
auth.set_access_token(access_token, access_token_secret) # access token and secret
api = tweepy.API(auth) # API object while passing in auth information

trends = api.trends_place(23424977)[0]['trends']
top_trends = sorted([trend['tweet_volume'] for trend in trends if
                     trend['tweet_volume']is not None], reverse=True)[:3]
tags = [trend['name'] for trend in trends if trend['tweet_volume'] in top_trends]
