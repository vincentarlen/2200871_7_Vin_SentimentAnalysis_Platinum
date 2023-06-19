import re
import pandas as pd

from flask import Flask, jsonify

app = Flask(__name__)

from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

##

df2 = pd.read_csv('new_kamusalay_ver3.csv', encoding='latin-1')
alay_mapping = dict(zip(df2['Alay'], df2['Correction']))
def cleaning_alay(text):
    wordlist = text.split()
    text_alay = [alay_mapping.get(x,x) for x in wordlist]
    clean_alay = ' '.join(text_alay)
    return clean_alay

df3 = pd.read_csv('abusive.csv', encoding='latin-1')
abusive_mapping = dict(zip(df3['ABUSIVE'], df3['SENSOR']))
def cleaning_abusive(text):
    wordlist_a = text.split()
    text_abusive = [abusive_mapping.get(x,x) for x in wordlist_a]
    clean_abusive = ' '.join(text_abusive)
    return clean_abusive

##

app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda: 'API Documentation for Data Processing and Modeling'),
    'version': LazyString(lambda: '1.0.0'),
    'description': LazyString(lambda: 'Dokumentasi API untuk Data Processing dan Modeling'),
    },
    host = LazyString(lambda: request.host)
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

@swag_from("docs/hello_world.yml", methods=['GET'])
@app.route('/', methods=['GET'])
def hello_world():
    json_response = {
        'status_code': 200,
        'description': "Menyapa Hello World",
        'data': "Hello World",
    }

    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/text.yml", methods=['GET'])
@app.route('/text', methods=['GET'])
def text():
    json_response = {
        'status_code': 200,
        'description': "Original Teks",
        'data': "Halo, apa kabar semua?",
    }

    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/text_clean.yml", methods=['GET'])
@app.route('/text-clean', methods=['GET'])
def text_clean():
    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah dibersihkan",
        'data': re.sub(r'[^a-zA-Z0-9]', ' ', "Halo, apa kabar semua?"),
    }

    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/text_processing.yml", methods=['POST'])
@app.route('/text-processing', methods=['POST'])
def text_processing():

    text = request.form.get('text')
#1 Regex Tanda Baca
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'@', ' ', text)
    text = re.sub(r'http:\S+', ' ', text)
#2 Regex Tambahan    
    text = re.sub(r'\\xa0|\\xa1|\\xa2|\\xa3|\\xa4|\\xa5|\\xa6|\\xa7', ' ', text)
    text = re.sub(r'\\xb0|\\xb1|\\xb2|\\xb3|\\xb4|\\xb5|\\xb6|\\xb7|\\xb8|\\xb9', ' ', text)
    text = re.sub(r'\\xc0|\\xc1|\\xc2|\\xc3|\\xc4|\\xc5|\\xc6', ' ', text)
    text = re.sub(r'\\xd0|\\xd1|\\xd2|\\xd3|\\xd4|\\xd5|\\xd6|\\xd7|\\xd8|\\xd9', ' ', text)
    text = re.sub(r'\\xe0|\\xe1|\\xe2|\\xe3|\\xe4|\\xe5|\\xe8', ' ', text)
    text = re.sub(r'\\x8a|\\x8b|\\x8c|\\x8d|\\x8e|\\x8f', ' ', text)
    text = re.sub(r'\\xaa|\\xab|\\xad|\\xaf', ' ', text)
    text = re.sub(r'\\xba|\\xbb|\\xbc|\\xbd|\\xbe', ' ', text)
    text = re.sub(r'\\xb1|\\xb5|\\xb7', ' ', text)
    text = re.sub(r'\\xca|\\xcb|\\xcc|\\xcd|\\xcf|\\xce', ' ', text)
    text = re.sub(r'\\xef', ' ', text)
    text = re.sub(r'\\x80|\\x81|\\x82|\\x83|\\x84|\\x85|\\x86|\\x87|\\x88|\\x89|\\x90|\\x91|\\x92|\\x93', ' ', text)
    text = re.sub(r'\\x9f', ' ', text)
#3 Regex Twitter
    text = re.sub('USER', r'', text)
    text = re.sub('User', r'', text)
    text = re.sub('user', r'', text)
    text = re.sub('RT', r'', text)
    text = re.sub('rt', r'', text)
    text = re.sub('URL', r'', text)
    text = re.sub('url', r'', text)
    text = re.sub('  ', ' ', text)
#4 Regex Angka    
    text = re.sub('0','nol ', text)
    text = re.sub('1','satu ', text)
    text = re.sub('2','dua ', text)
    text = re.sub('3','tiga ', text)
    text = re.sub('4','empat ', text)
    text = re.sub('5','lima ', text)
    text = re.sub('6','enam ', text)
    text = re.sub('7','tujuh ', text)
    text = re.sub('8','delapan ', text)
    text = re.sub('9','sembilan ', text)
    text = str(text).lower()
#5 Cleansing Alay
    text = cleaning_alay(text)
    text = cleaning_abusive(text)

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': text
    }

    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/text_processing_file.yml", methods=['POST'])
@app.route('/text-processing-file', methods=['POST'])
def text_processing_file():

        # Upladed file
    file = request.files.getlist('file')[0]

    # Import file csv ke Pandas
    df = pd.read_csv(file)

    # Ambil teks yang akan diproses dalam format list
    texts = df.text.to_list()

    # Lakukan cleansing pada teks
    cleaned_text = []
    for text in texts:
        def input(text):
        #1 Regex Tanda Baca
            text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
            text = re.sub(r'#\S+', ' ', text)
            text = re.sub(r'@\S+', ' ', text)
            text = re.sub(r'@', ' ', text)
            text = re.sub(r'http:\S+', ' ', text)
        #2 Regex Tambahan    
            text = re.sub(r'\\xa0|\\xa1|\\xa2|\\xa3|\\xa4|\\xa5|\\xa6|\\xa7', ' ', text)
            text = re.sub(r'\\xb0|\\xb1|\\xb2|\\xb3|\\xb4|\\xb5|\\xb6|\\xb7|\\xb8|\\xb9', ' ', text)
            text = re.sub(r'\\xc0|\\xc1|\\xc2|\\xc3|\\xc4|\\xc5|\\xc6', ' ', text)
            text = re.sub(r'\\xd0|\\xd1|\\xd2|\\xd3|\\xd4|\\xd5|\\xd6|\\xd7|\\xd8|\\xd9', ' ', text)
            text = re.sub(r'\\xe0|\\xe1|\\xe2|\\xe3|\\xe4|\\xe5|\\xe8', ' ', text)
            text = re.sub(r'\\x8a|\\x8b|\\x8c|\\x8d|\\x8e|\\x8f', ' ', text)
            text = re.sub(r'\\xaa|\\xab|\\xad|\\xaf', ' ', text)
            text = re.sub(r'\\xba|\\xbb|\\xbc|\\xbd|\\xbe', ' ', text)
            text = re.sub(r'\\xb1|\\xb5|\\xb7', ' ', text)
            text = re.sub(r'\\xca|\\xcb|\\xcc|\\xcd|\\xcf|\\xce', ' ', text)
            text = re.sub(r'\\xef', ' ', text)
            text = re.sub(r'\\x80|\\x81|\\x82|\\x83|\\x84|\\x85|\\x86|\\x87|\\x88|\\x89|\\x90|\\x91|\\x92|\\x93', ' ', text)
            text = re.sub(r'\\x9f', ' ', text)
        #3 Regex Twitter
            text = re.sub('USER', r'', text)
            text = re.sub('User', r'', text)
            text = re.sub('user', r'', text)
            text = re.sub('RT', r'', text)
            text = re.sub('rt', r'', text)
            text = re.sub('URL', r'', text)
            text = re.sub('url', r'', text)
            text = re.sub('  ', ' ', text)
        #4 Regex Angka    
            text = re.sub('0','nol ', text)
            text = re.sub('1','satu ', text)
            text = re.sub('2','dua ', text)
            text = re.sub('3','tiga ', text)
            text = re.sub('4','empat ', text)
            text = re.sub('5','lima ', text)
            text = re.sub('6','enam ', text)
            text = re.sub('7','tujuh ', text)
            text = re.sub('8','delapan ', text)
            text = re.sub('9','sembilan ', text)
            text = str(text).lower()
            text = cleaning_alay(text)
            text = cleaning_abusive(text)
            return text

        cleaned_text.append(input(text)),

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': cleaned_text,
    }

    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
   app.run()


