import tweepy


# ------------- set auth and initialize api -------------------------------
consumer_key = 'vv9NhyyyMTePWDBbNAkk7xly9'
consumer_secret = 'jRt3H4kIl2PAZQyV1m9PpcW96z1ncIVp68r2Dd1p94SyeWrm2t'
access_token = '948768906344312832-8DJ8eIZn01oNPIGgYBSEUeTu9azUGi0'
access_token_secret = 'uJ90fHHgYiyiWWDVwLkEbxQTnxzAXTmoHoJd8uHNA3tf9'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret) # authentication object
auth.set_access_token(access_token, access_token_secret) # access token and secret
api = tweepy.API(auth) # API object while passing in auth information

trends = api.trends_place(23424977)[0]['trends']
top_trends = sorted([trend['tweet_volume'] for trend in trends if
                     trend['tweet_volume']is not None], reverse=True)[:3]
tags = [trend['name'] for trend in trends if trend['tweet_volume'] in top_trends]
