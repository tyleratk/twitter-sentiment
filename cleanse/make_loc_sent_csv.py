import gmplot
import pickle
import pandas as pd
import re
import reverse_geocoder as rg

with open('../data/clean_tweets.pkl', 'rb') as infile:
    df = pickle.load(infile)

def get_coord(row):
    if row is None:
        return None
    nums = re.sub('[\[\]]', '', row).split(',')
    return [float(num) for num in nums]
    
def get_county(row):
    county = rg.search(row)[0]['admin2']
    return county 


df = df[['coordinates', 'sentiment', 'text']]
df.coordinates = df.coordinates.apply(get_coord)
df.dropna(axis=0, inplace=True)
df['county'] = df.coordinates.apply(get_county)

df.to_csv('county_sentiment_w_text.csv')



