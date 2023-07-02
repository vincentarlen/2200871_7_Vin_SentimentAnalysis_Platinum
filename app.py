import pandas as pd
from flask import Flask, jsonify, request
from flasgger import LazyJSONEncoder, LazyString, Swagger, swag_from

import cleansing
import nn

app = Flask(__name__)

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
    df['predicted_sentiment'] = df['text_preprocessed'].apply(nn.predict_sentiment)

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

    text = cleansing.preprocessing( request.form.get('text'))    

    # TODO: ADD MODEL & PREDICT 
    json_response = {
        'status_code': 200,
        'description': "Prediksi Sentimen",
        'data': text,
    }

    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/file_lstm.yml", methods=['POST'])
@app.route('/file-lstm', methods=['POST'])
def pred_file_lstm():

    # Uploaded file
    file = request.files.getlist('file')[0]

    # Import file csv ke Pandas
    df = pd.read_csv(file, encoding="latin-1")

    # Ambil teks yang akan diproses dalam format list
    texts = df["text"].to_list()

    # Lakukan cleansing pada teks
    cleaned_text = []
    for text in texts:
        cleaned_text.append(cleansing.preprocessing(text))

    # TODO: ADD MODEL & PREDICT 

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': cleaned_text,
    }

    response_data = jsonify(json_response)
    return response_data


if __name__ == '__main__':
    app.run()
