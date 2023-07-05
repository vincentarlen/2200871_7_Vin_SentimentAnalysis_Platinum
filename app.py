import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flasgger import LazyJSONEncoder, LazyString, Swagger, swag_from
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import cleansing 
import cleansing_NoStopword
import nn

app = Flask(__name__)

max_features = 100000
tokenizer = Tokenizer(num_words=max_features,split=' ')

sentiment_label = ['negative','neutral','positive']

with open('resource/model_lstm/bow/x_pad_sequences.pickle','rb') as handle:
    padded_sequences = pickle.load(handle)

model_lstm = load_model('resource/model_lstm/bow/model.h5')

app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info={
        'title': LazyString(lambda: 'DSC 7 Challenge Platinum Kelompok 1'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda: 'Dokumentasi API Sentiment analysis untuk challenge Binar platinum'),
    },
    host=LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,
                  config=swagger_config)


@app.route('/', methods=['GET'])
def hello_world():
    json_response = {
        'status_code': 200,
        'description': "/docs untuk dokumentasi",
        'data': "Hello",
    }

    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/text_nn.yml", methods=['POST'])
@app.route('/text-nn', methods=['POST'])
def pred_text_nn():

    text = cleansing.preprocessing(request.form.get('text'))

    pred = nn.predict_sentiment(text)

    json_response = {
        'status_code': 200,
        'description': 'Prediksi Sentimen untuk ' + request.form.get('text'),
        'data': pred,
    }

    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/file_nn.yml", methods=['POST'])
@app.route('/file-nn', methods=['POST'])
def pred_file_nn():

    # Uploaded file
    file = request.files.getlist('file')[0]

    # Import file csv ke Pandas
    df = pd.read_csv(file, encoding="latin-1")

    # Lakukan cleansing pada teks
    df['text_preprocessed'] = df['text'].apply(cleansing.preprocessing)
    # predict
    df['predicted_sentiment'] = df['text_preprocessed'].apply(
        nn.predict_sentiment)

    pred = list(zip(df['text'], df['predicted_sentiment']))

    json_response = {
        'status_code': 200,
        'description': "Prediksi Sentimen",
        'data': pred,
    }

    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/text_lstm.yml", methods=['POST'])
@app.route('/text-lstm', methods=['POST'])
def pred_text_lstm():

    real_text = request.form.get('text')
    text = cleansing_NoStopword.preprocessing(real_text)   
    #predict
    print(model_lstm.summary())
    tokenizer.fit_on_texts(text)
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature,maxlen=padded_sequences.shape[1])
    predict = model_lstm.predict(feature)
    sentiment = sentiment_label[np.argmax(predict[0])]

    json_response = {
        'status_code': 200,
        'description': "Prediksi Sentimen",
        'data': {
            'text': real_text,
            'sentiment':sentiment
        },
    }

    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/file_lstm.yml", methods=['POST'])
@app.route('/file-lstm', methods=['POST'])
def pred_file_lstm():

    #upload file
    file = request.files.getlist('file')[0]

    # Import file csv ke Pandas
    df = pd.read_csv(file, encoding='ISO-8859-1', delimiter="\t", names=['text', 'sentiment'])

    # Ambil teks yang akan diproses dalam format list
    texts = df["text"].apply(cleansing_NoStopword.preprocessing).tolist()

    # Lakukan cleansing pada teks
    tokenizer.fit_on_texts(texts)
    features = tokenizer.texts_to_sequences(texts)
    features = pad_sequences(features, maxlen=padded_sequences.shape[1])

    # Reshape the input data to match the model's expected shape
    features = np.expand_dims(features, axis=1)

    # Prediksi sentimen menggunakan model LSTM
    predicts = model_lstm.predict(features)
    sentiments = [sentiment_label[np.argmax(pred)] for pred in predicts]

    # Gabungkan teks awal dengan hasil prediksi
    result = list(zip(df["text"], sentiments))

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': result,
    }

    response_data = jsonify(json_response)
    return response_data


if __name__ == '__main__':
    app.run()
