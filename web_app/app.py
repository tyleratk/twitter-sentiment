from flask import Flask, render_template, request, make_response
from flask_table import Table, Col
from scripts.user_info import get_info
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import base64
import io
import pickle
import sys
sys.path.insert(0, '/Users/galvanize/twitter/src')
# sys.path.insert(0, '/home/ubuntu/twitter/src')
from new_model import TwitterClassifier

with open('static/models/model.pkl', 'rb') as infile:
    model = pickle.load(infile)

app = Flask(__name__)  


class ItemTable(Table):
    name = Col('Name')
    description = Col('Description')
    sentiment = Col('Sentiment')

class Item(object):
    def __init__(self, name, description, sentiment):
        self.name = name
        self.description = description
        self.sentiment = sentiment


@app.route('/get_user_info', methods = ["GET", "POST"])
def get_user_info():
    pd.set_option('display.max_colwidth', -1)

    user_name = request.form['user_name']
    option = request.form.get('option')

    if option == 'Search by User-name':
        try:
            results = get_info(user_name, option=option)
            df = pd.DataFrame(results, columns=['text', 'sentiment']).round(2)
            ten_most_recent = df.head(10)
            most_recent_preds = pd.DataFrame(model.predict(ten_most_recent.text.values),
                                             columns=["My Model's Prediction"])
            ten_most_recent = pd.concat([ten_most_recent, most_recent_preds], axis=1)
            preds = pd.DataFrame(model.predict(df.text.values), columns=["My Model's Prediction"]) 
            df = pd.concat([df, preds], axis=1).sort_values('sentiment', ascending=False)
            
            ten_most_recent = ten_most_recent.head(10).to_html(classes='most_recent')
            top10 = df.head(10).to_html(classes='most_recent')
            bottom10 = df.tail(10).sort_values('sentiment', ascending=False).to_html(classes='most_recent')
           
            return render_template('results.html', tables=[ten_most_recent,
                                                           top10, bottom10],
                                                   titles=['10 Most Recent',
                                                           'Top 10',
                                                           'Bottom 10'])
        except:
            return render_template('results.html', tables=['Please go back and try again'], 
                                   titles=['Something went wrong...'])

    elif option == 'Search by Hashtag':
        try:
            df = get_info(user_name, option=option).round(2)
            # histogram
            fig, ax = plt.subplots()
            title = 'Tweets for #{} Total of {} Tweets'.format(
                user_name, df.shape[0])
            sns.distplot(df.sentiment, ax=ax, label=title)
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            source = 'data:image/png;base64,{}'.format(plot_url)
            
            # top 10 pos/neg
            #
            preds = pd.DataFrame(model.predict(df.text.values), columns=["My Model's Prediction"])
            df = pd.concat([df, preds], axis=1).sort_values('sentiment', ascending=False)
            
            top10 = df.head(10).to_html(classes='most_recent')
            bottom10 = df.tail(10).sort_values('sentiment', ascending=False)
            bottom10 = bottom10.to_html(classes='most_recent')
            #, bottom10 , 'Bottom 10'
            return render_template('results.html', tables=[top10, bottom10],
                                   titles=['Top 10', 'Bottom 10'], plot=source,
                                   title=title)
        except:
            return render_template('results.html', tables=['Please go back and try again'], 
                                   titles=['Something went wrong...'])
    
    

@app.route('/')
def index():
    trending=['test', 'ttest','test']
    return render_template('index.html', hashtags=trending)
    

@app.route('/demo')
def demo():
    options = ['Search by User-name', 'Search by Hashtag']
    return render_template('demo.html', options=options)
    
    
@app.route('/about')
def about():
    # trending = ['test', 'test', 'test']
    trending='test'
    return render_template('about.html', hashtags=trending)
    
    
    
if __name__ == '__main__':
    app.run('0.0.0.0', port=8000, debug=True)
    # app.run('0.0.0.0', port=8105, threaded=True)

