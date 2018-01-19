import pickle
from string import punctuation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import textacy
import matplotlib.pyplot as plt
import networkx as nx
import spacy
nlp = spacy.load('en')



    
with open('../data/clean_tweets.pkl', 'rb') as infile:
    df = pickle.load(infile)

group = df.sample(100).text.values
corpus = ' '.join([doc for doc in group])

cleaned_text = textacy.preprocess_text(corpus, fix_unicode=True, no_accents=True)
doc = textacy.Doc(cleaned_text, lang='en')

graph = doc.to_semantic_network(nodes='words', edge_weighting='cooc_freq', window_width=2)
drop_nodes = [textacy.spacy_utils.normalized_str(tok) for tok in doc.tokens if tok.pos_ !='NOUN']
for node in drop_nodes:
    try:
        graph.remove_node(node)
    except:
        pass

node_weights = nx.pagerank_scipy(graph)
ax = textacy.viz.network.draw_semantic_network(graph, node_weights=node_weights, spread=2.5)
plt.show(block=False)








