from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pickle
import spacy
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import numpy as np

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

vectorizer = CountVectorizer(stop_words='english', min_df=5)
# tfidf_vectorizer = TfidfVectorizer()
# --- here
dtm = vectorizer.fit_transform(corpus).toarray()
vocab = np.array(vectorizer.get_feature_names())
num_topics = 2
num_top_words = 5
clf = NMF(n_components=num_topics, random_state=1)
doctopic = clf.fit_transform(dtm)

topic_words = []

for topic in clf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])
# doctopic = doctopic / np.sum(doctopic, axis=0, keepdims=True)
# tfidf = tfidf_vectorizer.fit_transform(docs)
# tfidf_feature_names = tfidf_vectorizer.get_feature_names()
# nmf = NMF(n_components=2).fit(tfidf)
# 
# topic_pr = nmf.transform(tfidf)
# 
# probs = abs(topic_pr) / abs(topic_pr.sum(axis=1))
# lda = LatentDirichletAllocation(n_components=2, max_iter=5)
# topic_pr = lda.fit_transform(tfidf)

