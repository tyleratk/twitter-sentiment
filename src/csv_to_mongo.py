import pandas as pd
import pymongo
import json
import clean_tweets



def get_data(filename):
    names = ['created_at', 'hash_tags', 'coordinates',
             'coordinates_type', 'lang', 'country', 'tweet_location', 'text',
             'user_created', 'default_profile_image', 'user_likes',
             'user_followers', 'user_following', 'user_screen_name',
             'user_num_tweets', 'user_location']
    data = pd.read_csv(filepath, names=names)
    data['text'] = data['text'].apply(clean_tweets.clean_text)
    return data


if __name__ == '__main__':
    filepath = '../data/tweets.csv'

    client = pymongo.MongoClient()
    db = client['tweet_data']
    table = db['tweets']
    table.drop()
    
    data = get_data(filepath)

    print('Getting ready to iterate...')
    for i in range(data.shape[0]):
        table.insert_one(json.loads(data.iloc[i].to_json()))
