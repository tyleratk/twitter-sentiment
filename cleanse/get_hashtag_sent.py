import pickle
import pandas as pd
from textacy.vsm import Vectorizer
import textacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
import spacy
from string import punctuation

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

nlp = spacy.load('en')
STOPLIST = set(list(ENGLISH_STOP_WORDS) + ["n't", "'s", "'m", "ca", "'", "'re",
                                           "pron"])

PUNCT_DICT = {ord(punc): None for punc in punctuation if punc not in ['_', '*']}


def clean_tweet(tweet):
    doc = nlp(tweet)
    doc = [word for word in doc if len(word) >= 4]
    pos_lst = ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']
    tokens = [token.lemma_.lower().replace(' ', '_') for token in doc if token.pos_ in pos_lst]
    return(' '.join(token for token in tokens if token not in STOPLIST).replace("'s", '').translate(PUNCT_DICT))


with open('../data/clean_tweets_hashtags.pkl', 'rb') as infile:
    df = pickle.load(infile)
    
hash_df = df[df['hash_tags'].str.contains('denver')]
print('Mean setiment score: {:.2f}'.format(hash_df.sentiment.mean()))
gb = hash_df.groupby('sentiment_type')
pos = gb.get_group('pos').text.values
neg = gb.get_group('neu').text.values

# for group in [pos, neg]:
#     corpus = [clean_tweet(tweet) for tweet in group]
# 
#     tf = TfidfVectorizer()
#     tf.fit(corpus)
# 
#     doc_term_matrix = tf.transform(corpus)
#     vocab = tf.get_feature_names()
#     vocab = [word for word in vocab if word != 'pron']
# 
#     model = textacy.tm.TopicModel('nmf', n_topics=5)
#     model.fit(doc_term_matrix)
#     # model.model
# 
#     for topic_idx, top_terms in model.top_topic_terms(vocab, top_n=10):
#         print('topic {}:   {}'.format(topic_idx, '   '.join(top_terms)))
# 
#     model.termite_plot(doc_term_matrix, vocab, topics=-1,
#                        n_terms=25, sort_terms_by='seriation', rank_terms_by='topic_weight',
#                        highlight_topics=None)
# plt.show(block=False)





