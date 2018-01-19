import pickle
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print('Topic {}:'.format(topic_idx+1))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print('\n')





if __name__ == '__main__':
    
    # with open('../data/clean_tweets_hashtags.pkl', 'rb') as infile:
    #     df = pickle.load(infile)
    with open('../data/clean_tweets_hashtags.pkl', 'rb') as infile:
        df = pickle.load(infile)

    hashtag_corpus = df.hash_tags.values
    
    hashtag_corpus = list(set([word for hashtag in hashtag_corpus for word in hashtag.split()]))
    with open('../data/hashtag_corpus.pkl', 'wb') as outfile:
        pickle.dump(hashtag_corpus, outfile)
    print('Wrote corpus file')

# ----------------------------------- LDA ----------------------------------------    
    # n_features = None
    # n_topics = 25
    # n_top_words = 10
    # 
    # vectorizer = CountVectorizer(min_df=2)
    # v = vectorizer.fit_transform(hashtag_corpus)
    # print("features ready.")
    # 
    # print("Fitting LDA models with tf features")
    # 
    # lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10,
    #                                 learning_method='online',
    #                                 learning_offset=50.,
    #                                 random_state=0)
    # lda.fit(v)
    # print("\nTopics in LDA model:")
    # feature_names = vectorizer.get_feature_names()
    # print_top_words(lda, feature_names, 10)










#--------------------------------- NMF -------------------------------------------
    # tfidf_vectorizer = TfidfVectorizer(min_df=2)
    # tfidf = tfidf_vectorizer.fit_transform(hashtag_corpus)
    # nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
    # 
    # print("\nTopics in NMF model:")
    # tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    # print_top_words(nmf, tfidf_feature_names, n_top_words)
