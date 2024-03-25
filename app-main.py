# Importing Library
import os
import flask
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

import pickle, re
import numpy as np
import sqlite3
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model

app = Flask(__name__)
app.json_encoder = LazyJSONEncoder

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "API Documentation for Sentiment Analysis",
        "description": "Dokumentasi API untuk Analisa Sentimen",
        "version": "1.0.0"
    }
}

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json'
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)


max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ',lower=True)
sentiment = ['negative', 'neutral', 'positive']

def cleansing(text):
    # Make sentence being lowercase
    text = text.lower()

    # Remove user, rt, \n, retweet, \t, url, xd
    pattern_1 = r'(user|retweet|\\t|\\r|url|xd)'
    text = re.sub(pattern_1, '', text)

    # Remove mention
    pattern_2 = r'@[^\s]+'
    text = re.sub(pattern_2, '', text)

    # Remove hashtag
    pattern_3 = r'#([^\s]+)'
    text = re.sub(pattern_3, '', text)

    # Remove general punctuation, math operation char, etc.
    pattern_4 = r'[\,\@\*\_\-\!\:\;\?\'\.\"\)\(\{\}\<\>\+\%\$\^\#\/\`\~\|\&\|]'
    text = re.sub(pattern_4, ' ', text)

    # Remove single character
    pattern_5 = r'\b\w{1,1}\b'
    text = re.sub(pattern_5, '', text)

    # Remove emoji
    pattern_6 = r'\\[a-z0-9]{1,5}'
    text = re.sub(pattern_6, '', text)

    # Remove digit character
    pattern_7 = r'\d+'
    text = re.sub(pattern_7, '', text)

    # Remove url start with http or https
    pattern_8 = r'(https|https:)'
    text = re.sub(pattern_8, '', text)

    # Remove (\); ([); (])
    pattern_9 = r'[\\\]\[]'
    text = re.sub(pattern_9, '', text)

    # Remove character non ASCII
    pattern_10 = r'[^\x00-\x7f]'
    text = re.sub(pattern_10, '', text)

    # Remove character non ASCII
    pattern_11 = r'(\\u[0-9A-Fa-f]+)'
    text = re.sub(pattern_11, '', text)

    # Remove multiple whitespace
    pattern_12 = r'(\s+|\\n)'
    text = re.sub(pattern_12, ' ', text)
    
    # Remove whitespace at the first and end sentences
    text = text.rstrip()
    text = text.lstrip()

    # Pattern to look for three or more repetitions of any character, including newlines.
    pattern_13 = re.compile(r"(.)\1{1,}", re.DOTALL)
    text = re.sub(pattern_13, r"\1\1", text)
    return text

db = sqlite3.connect('data.db', check_same_thread = False)
q_kamusalay = 'SELECT * FROM kamusalay'
t_kamusalay = pd.read_sql_query(q_kamusalay, db)
alay_dict = dict(zip(t_kamusalay['kata_alay'], t_kamusalay['normal']))

def alay_to_normal(text):
    for word in alay_dict:
        return ' '.join([alay_dict[word] if word in alay_dict else word for word in text.split(' ')])

def cleansingnew(sent):
    string = cleansing(sent)
    string = alay_to_normal(string)
    return string

# Load feature of neural network
#file = open('feature.p', 'rb')
#feature_file_from_nn = pickle.load(file)
#file.close()

# Load model of neural network
#model_file_from_nn = load_model('model_nn.p')

# Endpoint Neural Network Teks
# @swag_from("templates/NN_Text.yml",methods=['POST'])
# @app.route('/NN_Text',methods=['POST'])
# def nn_text():

#    original_text = request.form.get('text')
#    text = [cleansingnew(original_text)]

# Load file sequences lstm
file = open('resources_of_lstm/x_pad_sequences.pickle','rb')
feature_file_from_lstm = pickle.load(file)
file.close()

# Load Model LSTM
model_file_from_lstm = load_model('model_of_lstm/model.h5')

@swag_from("docs/LSTM_Text.yml",methods=['POST'])
@app.route('/LSTM_Text',methods=['POST'])
def lstm_text():

    original_text = request.form.get('text')

    text = [cleansingnew(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    guess = pad_sequences(feature,maxlen=feature_file_from_lstm.shape[1])

    prediction = model_file_from_lstm.predict(guess)
    polarity = np.argmax(prediction[0])
    get_sentiment = sentiment[polarity]

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using LSTM",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# Endpoint LSTM file
@swag_from("templates/LSTM_File.yml",methods=['POST'])
@app.route('/LSTM_File',methods=['POST'])
def lstm_file():
    file = request.files["upload_file"]
    df = (pd.read_csv(file, encoding="latin-1"))
    df = df.rename(columns={df.columns[0]: 'text'})
    df['text_clean'] = df.apply(lambda row : cleansingnew(row['text']), axis = 1)
    
    result = []

    for index, row in df.iterrows():
        text = tokenizer.texts_to_sequences([(row['text_clean'])])
        guess = pad_sequences(text, maxlen=feature_file_from_lstm.shape[1])
        prediction = model_file_from_lstm.predict(guess)
        polarity = np.argmax(prediction[0])
        get_sentiment = sentiment[polarity]
        result.append(get_sentiment)

    original = df.text_clean.to_list()

    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using LSTM",
        'data' : {
            'text' : original,
            'sentiment' : result
        },
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()