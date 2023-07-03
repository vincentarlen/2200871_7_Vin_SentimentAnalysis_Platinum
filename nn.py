import pickle
from sklearn.preprocessing import LabelEncoder

# exported with scikit v1.3.0
with open('resource/model_nn/labelencoder.pkl', 'rb') as file:
    le = pickle.load(file)
with open('resource/model_nn/model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('resource/model_nn/vectorizer.pkl', 'rb') as file:
    count_vect = pickle.load(file)

def predict_sentiment(text):
    text_vec = count_vect.transform([text])
    sentiment = model.predict(text_vec)
    sentiment = le.inverse_transform(sentiment)
    return sentiment[0]
