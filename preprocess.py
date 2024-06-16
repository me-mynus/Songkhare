import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

nltk.download("stopwords")
nltk.download("wordnet")
stopwords = set(stopwords.words("english"))


def data_clean_stem(text):
    if not isinstance(text, str):
        return ""
    stemmer = PorterStemmer()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    # List comprehension
    return " ".join(text)


def data_clean_lemm(text):
    wnl = WordNetLemmatizer()
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    words = word_tokenize(text)
    words = [wnl.lemmatize(word) for word in words if word not in stopwords]
    cleaned_text = " ".join(words).strip()
    return cleaned_text


def text_cleaning_lstm(df, column, vocab_size, max_length):
    stemmer = PorterStemmer()
    corpus = []  # Store preprocessed text for encoding
    for text in df[column]:
        text = re.sub("[^a-zA-Z]", " ", text)
        text = text.lower()
        text = text.split()
        text = [stemmer.stem(word) for word in text if word not in stopwords]
        text = " ".join(text)
        corpus.append(text)
    one_hot_rep = [one_hot(input_text=word, n=vocab_size) for word in corpus]
    pad = pad_sequences(sequences=one_hot_rep, maxlen=max_length, padding="pre")
    return pad
