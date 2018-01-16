import time
import numpy as np
import pandas as pd
import spacy
from string import punctuation
nlp = spacy.load('en')
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

STOPLIST = set(list(ENGLISH_STOP_WORDS) + ["n't", "'s", "'m", "ca", "'", "'re",
                                           "pron"])
PUNCT_DICT = {ord(punc): None for punc in punctuation if punc not in ['_', '*']}

n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 10


def clean_tweet(tweet):
    doc = nlp(tweet)
    # Let's merge all of the proper entities
    # for ent in doc.ents:
    #     if ent.root.tag_ != 'DT':
    #         ent.merge(ent.root.tag_, ent.text, ent.label_)
    #     else:
    #         # Keep entities like 'the New York Times' from getting dropped
    #         ent.merge(ent[-1].tag_, ent.text, ent.label_)

    # Part's of speech to keep in the result
    pos_lst = ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB'] # NUM?
    tokens = [token.lemma_.lower().replace(' ', '_') for token in doc if token.pos_ in pos_lst]

    return(' '.join(token for token in tokens if token not in STOPLIST).replace("'s", '').translate(PUNCT_DICT))

def print_n_words(model, feature_names, n):
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #{}: ".format(topic_idx+1))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n - 1:-1]]))
    print()

                                                                                          
with open('../data/clean_tweets.pkl', 'rb') as infile:
    df = pickle.load(infile)

gb = df.groupby('sentiment_type')
pos_tweets = gb.get_group('pos').sample(1000).text.values
neg_tweets = gb.get_group('neg').sample(1000).text.values


for group in [pos_tweets, neg_tweets]:
    corpus = [clean_tweet(tweet) for tweet in group] 
    vectorizer = CountVectorizer(max_df=0.90, min_df=8,
                                    max_features=n_features,
                                    stop_words='english')
    v = vectorizer.fit_transform(corpus)
    print("features ready.")


    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(v)
    print("\nTopics in LDA model:")
    feature_names = vectorizer.get_feature_names()
    print_n_words(lda, feature_names, 10)









