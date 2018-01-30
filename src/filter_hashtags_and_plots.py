import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns

def get_topic(df, topic):
    df = df[df['text'].str.contains(topic, case=False) == True]
    return df
    
    
def get_cloud(df, group, title, save):
    gb = df.sort_values('sentiment').groupby('sentiment_type')

    stopwords = set(list(STOPWORDS))# + ['football', 'shit', 'fuck', 'ass'])
    # mpl.rcParams['font.size'] = 12                #10 
    mpl.rcParams['savefig.dpi'] = 100              #72 
    mpl.rcParams['figure.subplot.bottom'] = .1 
    wordcloud = WordCloud(
                          stopwords=stopwords,
                          background_color='white',
                          # color_map=
                          max_font_size=40,
                          random_state=42,
                          width=500,
                          height=250
                         ).generate(' '.join(gb.get_group(group).text))
    plt.figure(figsize=(20,10))
    plt.title(title)
    plt.imshow(wordcloud)
    plt.axis('off')
    if save:
        plt.savefig('../images/plots/cloud_' + group, facecolor='k',
                    bbox_inches='tight')
    else:
        plt.tight_layout(pad=0)
        plt.show(block=False)
        
        
def get_hist(df, title, save):
    # plt.figure(figsize=(20,10))
    sns.distplot(df.sentiment)
    plt.title(title)
    plt.show(block=False)
    if save:
        plt.savefig('../images/plots/hist_' + title)        
        
def get_plot(plot_type, data, keywords, cloud_group=None, title='', save=False):
    data = get_topic(data, keywords)
    
    if plot_type == 'hist':
        get_hist(data, title, save)
    elif plot_type == 'cloud':
        get_cloud(data, cloud_group, title, save)
    

if __name__ == '__main__':
    # with open('../data/clean_tweets.pkl', 'rb') as infile:
    #     df = pickle.load(infile)
    df = pd.read_csv('../data/super_bowl.csv', names=['data','coord',
                     'tweet_location','text','user_loc','sentiment'])
    df = df[['text', 'sentiment']]
    
    # -------------------- histograms ----------------------------------------         
    get_plot(plot_type='cloud', data=df, keywords='tom brady', save=False)
             #title='Filtering tweets by "eagles"', save=False)
            
    # ----------------------- word clouds ------------------------------------ 
    # get_plot(plot_type='cloud', data=df, keywords='georgia|alabama|football',
    #          cloud_group='pos', title='Filtering tweets by georgia, alabama and football',
    #          save=True)
    # get_plot(plot_type='cloud', data=df, keywords='georgia|alabama|football',
    #          cloud_group='neg', title='Filtering tweets by georgia, alabama and football',
    #          save=True)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    