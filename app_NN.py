from flask import Flask, jsonify
from flask import request
import flask
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

import pickle, re
import numpy as np
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
# from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

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




model_nn = pickle.load(open("model_nn.p", "rb"))
vect_nn = pickle.load(open("feature.p", "rb"))

def lowercase(s):
    return s.lower()

def remove_punctuation(s):
    s = re.sub('[^0-9a-zA-Z]+', ' ', s)
    s = re.sub(r':', '', s)
    s = re.sub('\n',' ',s)
    s = re.sub('rt',' ', s)
    s = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ', s)
    s = re.sub('  +', ' ', s)
    s = re.sub(r'pic.twitter.com.[\w]+', '', s)
    s = re.sub('user',' ', s)
    s = re.sub('gue','saya', s)
    s = re.sub(r'‚Ä¶', '', s)
    return s

db = sqlite3.connect('data.db', check_same_thread = False)
q_kamusalay = 'SELECT * FROM kamusalay'
t_kamusalay = pd.read_sql_query(q_kamusalay, db)
alay_dict = dict(zip(t_kamusalay['kata_alay'], t_kamusalay['normal']))

def alay_to_normal(s):
    for word in alay_dict:
        return ' '.join([alay_dict[word] if word in alay_dict else word for word in s.split(' ')])

def cleansing(sent):
    string = lowercase(sent)
    string = remove_punctuation(string)
    string = alay_to_normal(string)
    return string



# # Endpoint NN teks
@swag_from('docs/NN_Text.yml',methods=['POST'])
@app.route('/NN_Text',methods=['POST'])
def nn_text():

    original_text = request.form.get('text')

    text = vect_nn.transform[cleansing(original_text)]

    result = model_nn.predict(text)
    resultjson = result.tolist()[0]

    json_response = {
        'status_code': 200,
        'description': 'Result of Sentiment Analysis using RNN',
        'data_raw': text,
        'data_clean': resultjson
    }
    response_data = jsonify(json_response)
    return response_data

# # Endpoint NN file
# @swag_from('docs/NN_File.yml',methods=['POST'])
# @app.route('/NN_File',methods=['POST'])
# def nn_file():
#     file = request.files.getlist('file')[0]

#     df = pd.read_csv(file, encoding='ISO-8859-1')

    

if __name__ == '__main__':
    app.run()