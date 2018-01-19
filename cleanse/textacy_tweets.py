import pickle
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import textacy
import matplotlib.pyplot as plt
import networkx as nx
import spacy
from textacy.vsm import Vectorizer
nlp = spacy.load('en')
STOPLIST = set(list(ENGLISH_STOP_WORDS) + ["n't","'s","'m","ca","'","'re","pron"])
PUNCT_DICT = {ord(punc): None for punc in punctuation if punc not in ['_', '*']}



def clean_tweet(tweet):
    doc = nlp(tweet)
    doc = [word for word in doc if len(word) >= 4]
    pos_lst = ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']
    tokens = [token.lemma_.lower().replace(' ', '_') for token in doc if token.pos_ in pos_lst]

    return(' '.join(token for token in tokens if token not in STOPLIST).replace("'s", '').translate(PUNCT_DICT))
    
with open('../data/clean_tweets.pkl', 'rb') as infile:
    df = pickle.load(infile)

group = df.sample(100).text.values
corpus = [clean_tweet(tweet) for tweet in group] 


vectorizer = Vectorizer(weighting='tfidf', normalize=True, smooth_idf=True,
                        min_df=2, max_df=0.95, max_n_terms=100000)
doc_term_matrix = vectorizer.fit_transform(corpus)

model = textacy.tm.TopicModel('nmf', n_topics=5)
model.fit(doc_term_matrix)

doc_topic_matrix = model.transform(doc_term_matrix)
for topic_idx, top_terms in model.top_topic_terms(vectorizer.id_to_term, topics=[0,1]):
    print('topic', topic_idx, ':', '   '.join(top_terms))
