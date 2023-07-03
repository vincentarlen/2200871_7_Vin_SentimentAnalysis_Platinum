import re
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from keras.models import load_model
import time
import pickle 
# from imblearn.over_sampling import RandomOverSampler

start_time = time.time()

max_features = 100000
embedding_size = 100

def fix_word(text):
    return ' '.join([kamus_dict[word] if word in kamus_dict else word for word in text.split(' ')])

def remove_unnecessaryChar(text):
    text = re.sub(r'&amp;|amp;|&', 'dan', text)
    text = re.sub(r'\\n+', '', text)
    text = re.sub('&lt;/?[a-z]+&gt;', ' ', text)
    text = re.sub(r'#+','#', text)
    text = re.sub(r'http\S+',' ',text)
    text = re.sub(r'(USER+\s?|RT+\s?|URL+\s?)', ' ', text)
    text = re.sub(r'x[a-zA-Z0-9]+', ' ', text)
    return text

def remove_punctuation(text):
    text = re.sub(r'\?', '', text)
    text = re.sub(r'[^a-zA-Z0-9]+', ' ', text)
    text = re.sub(r' +', ' ', text.lower().lstrip("0123456789").strip())
    return text

def remove_stopwords(text):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    return stopword.remove(text)

def preprocessing(text):
    text = remove_unnecessaryChar(text)
    text = remove_punctuation(text)
    text = fix_word(text)
    # text = remove_stopwords(text)
    return text

## read and do preprocessing on the data
df = pd.read_csv('train_preprocess.tsv.txt', encoding='ISO-8859-1', delimiter="\t", names=['text','sentiment'])
df["text"] = df["text"].str.encode('ascii', 'ignore').str.decode('ascii')
df.drop_duplicates(inplace=True)
print(df.shape)
kamus = pd.read_csv('new_kamusalay.csv', names=['old','new'], encoding='ISO-8859-1')
kamus_dict = dict(zip(kamus['old'], kamus['new']))

df["text"] = df["text"].apply(preprocessing)
df.replace('', pd.NA, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

## check size each label
negative = df.loc[df['sentiment'] == 'negative'].text.tolist()
positive = df.loc[df['sentiment'] == 'positive'].text.tolist()
neutral = df.loc[df['sentiment'] == 'neutral'].text.tolist()

neg_label = df.loc[df['sentiment'] == 'negative'].sentiment.tolist()
pos_label = df.loc[df['sentiment'] == 'positive'].sentiment.tolist()
neut_label = df.loc[df['sentiment'] == 'neutral'].sentiment.tolist()

total_data = positive + negative + neutral
labels = pos_label + neg_label + neut_label
print("total data = %s" % len(total_data))

## tokenization using TF-IDF
tokenizer = Tokenizer(num_words=max_features,split=' ')
tokenizer.fit_on_texts(total_data)
with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print("tokenizer.pickle has created!")
X = tokenizer.texts_to_sequences(total_data)
# print(X)

vocab_size = len(tokenizer.word_index)
max_len = max(len(x) for x in X)

X = pad_sequences(X)
with open('x_pad_sequences.pickle','wb') as handle:
    pickle.dump(X,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print("x_pad_sequences.pickle has created!")

Y = pd.get_dummies(labels)
Y = Y.values

with open('y_labels.pickle','wb') as handle:
    pickle.dump(Y,handle,protocol=pickle.HIGHEST_PROTOCOL)
    print("y_labels.pickle has created!")

file = open("x_pad_sequences.pickle",'rb')
X = pickle.load(file)
file.close()

file = open("y_labels.pickle",'rb')
Y = pickle.load(file)
file.close()

# oversampler = RandomOverSampler(random_state=123)
# X_resampled, y_resampled = oversampler.fit_resample(X, Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

sentences = [text.split() for text in total_data]
word2vec_model = Word2Vec(sentences, vector_size=embedding_size, window=5, min_count=1, workers=4)
embedding_matrix = np.zeros((vocab_size+1, embedding_size))
for word, i in tokenizer.word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

model = Sequential()
model.add(Embedding(vocab_size+1, embedding_size, weights=[embedding_matrix], input_length=X.shape[1],trainable=True))
model.add(LSTM(64, dropout=0.2))
model.add(Dense(3, activation='softmax'))
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=64, callbacks=[early_stopping],verbose=1)
model.save('model.h5')
# Train the model

# model.load('model.h5')
prediction = model.predict(X_test)
y_pred = prediction
print(y_pred)

matrix_test = metrics.classification_report(y_test.argmax(axis=1),y_pred.argmax(axis=1))
print("done testing")
print(matrix_test)

train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Get training and validation accuracy values
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot loss values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot loss values
ax1.plot(train_loss, label='Training Loss')
ax1.plot(val_loss, label='Validation Loss')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot accuracy values
ax2.plot(train_accuracy, label='Training Accuracy')
ax2.plot(val_accuracy, label='Validation Accuracy')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()

end_time = time.time()
execution_time = end_time - start_time
print("Waktu yang dibutuhkan: ", execution_time, " detik")
