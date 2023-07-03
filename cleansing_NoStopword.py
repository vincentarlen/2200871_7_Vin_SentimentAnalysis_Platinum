import re
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory # pip install sastrawi

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

def preprocessing(text):
     text = remove_unnecessaryChar(text)
     text = remove_punctuation(text)
     text = fix_word(text)

     return text

df = pd.read_csv('train_preprocess.tsv.txt', encoding='ISO-8859-1', delimiter="\t", names=['text','sentiment'])
df["text"] = df["text"].str.encode('ascii', 'ignore').str.decode('ascii')
df.drop_duplicates()
# print(df.head(30))
kamus = pd.read_csv('new_kamusalay.csv', names=['old','new'], encoding='ISO-8859-1')
kamus_dict = dict(zip(kamus['old'], kamus['new']))

df["text"] = df["text"].apply(preprocessing)# apply cleansing 
df.replace('', pd.NA, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# print(df["text"].head(30))
# final = pd.DataFrame(df)
# final.to_csv('try.csv')

