import pickle
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import textacy
import matplotlib.pyplot as plt
import spacy
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
    
with open('../data/clean_tweets.pkl', 'rb') as infile:
    df = pickle.load(infile)

group = df.sample(1000).text.values
corpus = [clean_tweet(tweet) for tweet in group]
# group = ' '.join([tweet for tweet in group])
# corpus = textacy.preprocess_text(group, fix_unicode=True, lowercase=True)

tf = TfidfVectorizer().fit(corpus)
doc_term_matrix = tf.transform(corpus)
vocab = tf.get_feature_names()
vocab = [word for word in vocab if word != 'pron']

model = textacy.tm.TopicModel('nmf', n_topics=8)
model.fit(doc_term_matrix)
# model.model

for topic_idx, top_terms in model.top_topic_terms(vocab, top_n=10):
    print('topic {}:   {}'.format(topic_idx, '   '.join(top_terms)))
    
model.termite_plot(doc_term_matrix, vocab, topics=-1,
                   n_terms=25, sort_terms_by='seriation', rank_terms_by='topic_weight',
                   highlight_topics=None)
plt.show(block=False)









