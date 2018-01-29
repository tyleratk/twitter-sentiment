import tweepy
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scripts import hashtag_sentiment
from scripts import stream_hashtags

def get_api():
    # ------------- set auth and initialize api -------------------------------
    consumer_key = 'dSJjdS3K25Ff3wl8uqZFKFgIZ'
    consumer_secret = '0z92aeUboBEXOFgmqSX1D6FYhfBv4LE5L14zz4OtcRYsIp5TbJ'
    access_token = '948768906344312832-eCKxtrwvuZBfa92mKJ1TJTLvkfOmLec'
    access_token_secret = 'ERVheG8crqQO0wQ4mmKhcVA5DLroFP80uoKFMTJKPgFZR'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api, auth
    
    
def clean_text(tweet):
    '''
    remove links, usernames, and newlines from tweet
    (@\w+)               => removes usernames
    (#\w+)               => removes hashtags
    (\w+:\/\/\S+)        => removes links
    '''
    return re.sub('(@\w+)|(#\w+)|(\w+:\/\/\S+)|(\n+)', '', tweet)


def get_sentiment(text):
    text = clean_text(text)
    sid = SentimentIntensityAnalyzer()
    sent = sid.polarity_scores(text)['compound']
    return sent
    
    
def get_all_tweets(screen_name, api):
	'''
	gets all tweets from screen_name
	
	Input:
		screen_name: twitter user_name - string
		api: authenticated twitter api
	Output:
		list of json objects from 3240 most recent tweets or all, only allowed
		to get 3240 most recent tweets with this method
	'''
	alltweets = []	
	new_tweets = api.user_timeline(screen_name=screen_name, count=200,
                                   tweet_mode='extended')
	alltweets.extend(new_tweets)

	oldest = alltweets[-1].id - 1

	while len(new_tweets) > 0:
		print ("getting tweets before %s" % (oldest))
		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest, 
                                       tweet_mode='extended')
		alltweets.extend(new_tweets)
		oldest = alltweets[-1].id - 1
		print ("...%s tweets downloaded so far" % (len(alltweets)))
	return alltweets
    
    
def parse_tweets(api, username):
    tweets = get_all_tweets(username, api)
    results = []
    
    for tweet in tweets:
        try:
            tweet = tweet._json
        except:
            continue
        try:
            text = tweet['full_text']
        except:
            text = tweet['text']

        sentiment = get_sentiment(text)

        tweet_data = [text, sentiment]
        
        results.append(tweet_data)
    return results
    




# ------------------- MAIN FUNCTION HERE -------------------------------------

def get_info(keyword, option):
    api, auth = get_api()
    if option == 'Search by User-name':
        results = parse_tweets(api, keyword)
    elif option == 'Search by Hashtag':
        stream_hashtags.get_hashtag_info(auth, keyword)
        results = pd.read_csv('data/' + keyword + '.csv', names=['text', 'sentiment'])

    return results
    
    
if __name__ == '__main__':
    pass
    # api = get_api()
    # tweets = parse_tweets(api, 'realdonaldtrump') #









