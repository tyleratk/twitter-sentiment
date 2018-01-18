import gmplot
import pickle
import pandas as pd
import re

with open('../data/clean_tweets.pkl', 'rb') as infile:
    df = pickle.load(infile)

def get_coord(row):
    nums = re.sub('[\[\]]', '', row).split(',')
    return [float(num) for num in nums]
    
    
coord = df.coordinates.dropna()
coord = coord.apply(get_coord)
lat = [c[0] for c in coord]
lng = [c[1] for c in coord]
weights

gmap = gmplot.GoogleMapPlotter(34.0522, -118.2437, 16)

gmap.heatmap_weighted(lat, lng, weights)

map_styles = [
    {
        'featureType': 'all',
        'stylers': [
            {'saturation': -80 },
            {'lightness': 60 },
        ]
    }
]

gmap.draw("mymap.html", map_styles=map_styles)
    
gmap.heatmap(lat, lng)
gmap.draw("mymap.html")