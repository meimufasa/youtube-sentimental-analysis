import pandas as pd
import joblib
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji
import re
from deep_translator import GoogleTranslator
from nltk.data import find

# this code ensures nltk downloads will only be downloaded once
def ensure_nltk_resources():
    def safe_download(resource):
        try:
            find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1])

    for r in ['tokenizers/punkt', 'corpora/stopwords', 'corpora/wordnet', 'corpora/omw-1.4']:
        safe_download(r)


def contractions_fix(text):
    x=[]
    for i in text.split():
        x.append(contractions.fix(i))

    return " ".join(x)

def remove_stopwords(text):
    x=[]
    for i in text.split():
        if i not in stopwords.words('english'):
            x.append(i)

    return " ".join(x)

lemmatizer = WordNetLemmatizer()

def stemming(text):
    x=[]
    for i in text.split():
        x.append(lemmatizer.lemmatize(i))

    return " ".join(x)

def preprocessing_pipe(df):
  df = df.apply(contractions_fix)
  df = df.apply(remove_stopwords)
  df = df.apply(stemming)
  return df

pipe = joblib.load('model.pkl')

def prediction(X):
  X = convert_emojis(X)
  X  = translate_to_english(X)
  if X != pd.DataFrame:
    X = pd.Series(X)
  X = preprocessing_pipe(X)
  y_pred = pipe.predict(X)
  print(y_pred)
  return y_pred

def convert_emojis(X):
    if isinstance(X, str):
        X = [X]
    return [emoji.demojize(text).replace(":", "").replace("_", " ") for text in X]

def translate_to_english(X):
    translated_texts = []
    for text in X:
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            translated_texts.append(translated)
            print(translated_texts)
        except Exception as e:
            print(f"Translation error: {e}")
            translated_texts.append(text)
    return translated_texts