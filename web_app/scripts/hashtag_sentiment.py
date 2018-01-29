import pickle
import pandas as pd
# from textacy.vsm import Vectorizer
# import textacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
import spacy
import tweepy
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import os
import networkx as nx
import time


nlp = spacy.load('en')
STOPLIST = set(list(ENGLISH_STOP_WORDS) + ["n't", "'s", "'m", "ca", "'", "'re",
                                           "pron"])
PUNCT_DICT = {ord(punc): None for punc in punctuation if punc not in ['_', '*']}
# ------------- set auth and initialize api -------------------------------
consumer_key = os.environ.get('twitter_consumer_key')
consumer_secret = os.environ.get('twitter_consumer_secret')
access_token = os.environ.get('twitter_access_token')
access_token_secret = os.environ.get('twitter_token_secret')
# -------------------------------------------------------------------------

class HashtagSentiment():
    
    def __init__(self, query, lang='en', show_plot=True):
        self.query = query
        self.lang = lang
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)
        self.tweets = []
        self.search_twitter()
        self.show_plot = show_plot
        # self.tweet_count = 200
        
        
    def prep_tokens(self, tweet):
        doc = nlp(tweet)
        doc = [word for word in doc if len(word) >= 4]
        pos_lst = ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']
        tokens = [token.lemma_.lower().replace(' ', '_') for token in doc if token.pos_ in pos_lst]
        return(' '.join(token for token in tokens if token not in STOPLIST).replace("'s", '').translate(PUNCT_DICT))
        
        
    def show_plot(self, pos, neg, plot_type='semantic'):
        for group in [('Positive', pos), ('Negative', neg)]:
            fig = plt.figure()
            name, group = group
            if plot_type == 'semantic':
                # if name == 'Positive' and group.shape[0] > 150:
                #     group = group.sample(155)
                corpus = [self.prep_tokens(tweet) for tweet in group]
                corpus = ' '.join(word for word in corpus)
                cleaned_text = textacy.preprocess_text(corpus, fix_unicode=True,
                                                       no_accents=True)
                doc = textacy.Doc(cleaned_text, lang='en')
                graph = doc.to_semantic_network(nodes='words', edge_weighting='cooc_freq', 
                                                window_width=10)
                drop_nodes = ['pron']
                for node in drop_nodes:
                    try:
                        graph.remove_node(node)
                    except:
                        pass
                node_weights = nx.pagerank_scipy(graph)
                ax = textacy.viz.network.draw_semantic_network(graph, 
                                node_weights=node_weights, spread=50.0)
                plt.suptitle(name + ' Sentiment Topics:' + '\n{} {} tweets\n{}'.format(
                                group.shape[0], name, self.hashtag))
                # plt.savefig('../images/plots/' + name)
            else:
                corpus = [self.prep_tokens(tweet) for tweet in group]
                tf = TfidfVectorizer().fit(corpus)
            
                doc_term_matrix = tf.transform(corpus)
                vocab = tf.get_feature_names()
                vocab = [word for word in vocab if word != 'pron']
            
                model = textacy.tm.TopicModel('nmf', n_topics=3)
                model.fit(doc_term_matrix)            
                model.termite_plot(doc_term_matrix, vocab, topics=-1,
                                   n_terms=25, sort_terms_by='seriation',
                                   rank_terms_by='topic_weight',
                                   highlight_topics=range(3))
                plt.suptitle(name + ' Sentiment Topics:')
                # plt.savefig('semantic_plot')

        # plt.show(block=False)
        return plt

        
    def clean_text(self, tweet):
        '''
        remove links, usernames, and newlines from tweet
        (@\w+)               => removes usernames
        (#\w+)               => removes hashtags
        (\w+:\/\/\S+)        => removes links
        '''
        return re.sub('(@\w+)|(#\w+)|(\w+:\/\/\S+)|(\n+)', '', tweet)
        
        
    def get_sentiment(self):
        try:
            df = self.df
            df = pd.concat(df, pd.DataFrame(self.tweets, columns=['text']))
        except:
            df = pd.DataFrame(self.tweets, columns=['text'])
 
        df['text'] = df['text'].apply(self.clean_text)
        sid = SentimentIntensityAnalyzer()
        df['sentiment'] = df.text.apply(lambda x: sid.polarity_scores(x)['compound'])   
        df.loc[df.sentiment >  0.1,'sentiment_type'] = 'pos'
        mask = (df.sentiment >= -0.1) & (df.sentiment <= 0.1)
        df.loc[mask, 'sentiment_type'] = 'neu'
        df.loc[df.sentiment <  -0.1,'sentiment_type'] = 'neg'
        gb = df.groupby('sentiment_type')
        pos = gb.get_group('pos').text.values
        neg = gb.get_group('neg').text.values

        if df.shape[0] > 25:
        # if pos.shape[0] > 300 and neg.shape[0] > 300:
            hashtag = self.query
            self.hashtag = re.sub('%23', '#', hashtag)
            print('Mean sentiment score for "{}": {:.2f}'.format(hashtag, 
                  df.sentiment.mean()))
            print('There were {} positive tweets and {} negative tweets'.format(
                  pos.shape[0], neg.shape[0]))
            if self.show_plot:
                try:
                    self.show_plot(pos, neg)
                except:
                    self.search_twitter()
        else:
            time.sleep(1)
            self.search_twitter()
        
        
    def search_twitter(self):
        results = self.api.search(q=self.query, lang=self.lang, count=35,
                                  tweet_mode='extended')
        for tweet in results:
            tweet = tweet._json
            self.tweets.append(tweet['full_text'])
        self.get_sentiment()
    
            

if __name__ == '__main__':
    test = HashtagSentiment('%23OscarNoms')





