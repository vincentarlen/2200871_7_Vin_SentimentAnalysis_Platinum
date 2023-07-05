# Sentiment Analysis API - README

This repository contains code for a Sentiment Analysis API using Flask. The API utilizes pre-trained models to predict the sentiment of text inputs. It provides endpoints for both single text prediction and batch prediction on a file containing multiple texts.

## Requirements

- Python 3.x
- Flask
- Pandas
- NumPy
- TensorFlow
- Keras
- Flask-Swagger
- Flask-JSONEncoder

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sentiment-analysis-api.git

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt

## Usage

Start the Flask server:

   ```bash
   python app.py
   ```
The server will start running at http://localhost:5000.

Open your web browser and go to http://localhost:5000/docs to view the API documentation.

Use the available endpoints to make predictions:

- **/text-nn**: Predict sentiment for a single text using a neural network model.
- **/file-nn**: Predict sentiment for multiple texts in a file using a neural network model.
- **/text-lstm**: Predict sentiment for a single text using an LSTM model.
- **/file-lstm**: Predict sentiment for multiple texts in a file using an LSTM model.

Follow the API documentation for each endpoint to provide the required input parameters.

## Model and Data

The API uses pre-trained models for sentiment analysis. The models are stored in the **'resource/model_lstm'** and **'resource/model_nn'** directories. The dataset used to train the models is not included in this repository.

**NOTE**: IF USING **'model_lstm/word2vec'** or **'model_lstm/word2vec_NoStopWord'**, REMOVE/COMMENT LINE 119 AND 155 ON **'app.py'**
