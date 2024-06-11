import pandas as pd
import numpy as np
import pickle
import nltk
import re
import os
from nltk.stem import PorterStemmer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# LSTM Library
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import one_hot
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model

# ML models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# plt.style.use("ggplot")
train_df = pd.read_csv(
    "data/train.txt",
    header=None,
    sep=";",
    names=["text", "sentiment"],
    encoding="utf-8",
)
train_df["length"] = [len(x) for x in train_df.text]

test_df = pd.read_csv(
    "data/test.txt", header=None, sep=";", names=["text"], encoding="utf-8"
)

test_df["length"] = [len(x) for x in test_df.text]

# Preprocessing
# lb encodes the labels for the sentiments numerically
lb = LabelEncoder()
train_df["sentiment"] = lb.fit_transform(train_df["sentiment"])
df = train_df.copy()

# Data cleaning process

tqdm.pandas()
nltk.download("stopwords")
# stopwords are words with little to no meaning. Eg: is, the, a, an, etc.
stopwords = set(nltk.corpus.stopwords.words("english"))


# Stemmer: algorithm that reduces words to their base form(Stem). running reduced to run.
def data_clean(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    # List comprehension
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)


df["cleaned"] = df["text"].progress_apply(data_clean)

X_train, X_test, Y_train, Y_test = train_test_split(
    df["cleaned"], df["sentiment"], test_size=0.2, random_state=42
)

# Feature extraction
# tfid is a feature extraction technique that assigns a score to each word based on its importance in the document
tfidvec = TfidfVectorizer()
X_train_tf = tfidvec.fit_transform(X_train)
X_test_tf = tfidvec.transform(X_test)
# print(df.head())

# classifiers = {
#     "Naive Bayes": MultinomialNB(),
#     "Logistic Regression": LogisticRegression(),
#     "Random Forest": RandomForestClassifier(),
#     "Support Vector Machine": SVC(),
# }


# for name, clf in classifiers.items():
#     print(f"\n{name}\n")
#     clf.fit(X_train_tf, Y_train)
#     Y_pred_tf = clf.predict(X_test_tf)
#     acc = accuracy_score(Y_test, Y_pred_tf)
#     print(f"========{acc}========")
#     print("Classification Report")
#     print(classification_report(Y_test, Y_pred_tf))

# Classification report returns
# Precision: The ratio of correctly predicted positive observations to the total predicted positive observations (T/T+FP)
# Recall: The ratio of correctly predicted positive observations to the all observations in actual class (T/T+FN)
# F1 Score: Harmonic mean of precision aand recall

# lg = LogisticRegression(max_iter=1000)
# lg.fit(X_train_tf, Y_train)
# lg_y_predict = lg.predict(X_test_tf)


# def predict_emotion(input):
#     clean = data_clean(input)
#     input_vec = tfidvec.transform([clean])
#     predicted_label = lg.predict(input_vec)[0]
#     emotion = lb.inverse_transform([predicted_label])[0]
#     label = np.max(lg.predict(input_vec))

#     return emotion, label


# LG displays a very linear approach to emotion recognition. text below is clearly sadness but because it sees love, it predicts love.

# predicted_emotion = predict_emotion(
#     "I loved her so much, now I am just a shell of myself"
#     # "I love icecream very much, it makes me so happy"
# )
# print(
#     "Emotion predicted is:",
#     predicted_emotion[0],
#     "with the label designated as:",
#     predicted_emotion[1],
# )

# folder = "model_data"

# if not os.path.exists(folder):
#     os.makedirs(folder)

# if not os.listdir(folder):
#     pickle.dump(lg, open("model_data/lg_model.pkl", "wb"))
#     pickle.dump(lb, open("model_data/label_encoder.pkl", "wb"))
#     pickle.dump(tfidvec, open("model_data/tf_vectorizer.pkl", "wb"))


# Using LSTM Model for emotion recognition
# For a deep learning model, one hot encoding and padding is required


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


x_train_lstm = text_cleaning_lstm(train_df, "text", 11000, 300)
y_train_lstm = to_categorical(train_df["sentiment"])


# def model_train(x, y):
#     model = Sequential()
#     model.add(Embedding(input_dim=11000, output_dim=150, input_length=300))
#     model.add(Dropout(0.2))
#     model.add(LSTM(128))
#     model.add(Dropout(0.2))
#     model.add(Dense(64, activation="sigmoid"))
#     model.add(Dropout(0.2))
#     model.add(Dense(6, activation="softmax"))
#     model.compile(
#         optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
#     )

#     callback = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

#     model.fit(
#         x,
#         y,
#         verbose=1,
#         epochs=30,
#         batch_size=32,
#         callbacks=[callback],
#     )
#     model.save("model_data/lstm_model_2.h5")
#     return model


# model = model_train(x_train_lstm, y_train_lstm)

model = load_model("model_data/lstm_model_2.h5")


def predict_emotion_lstm(input):
    stemmer = PorterStemmer()
    corpus = []
    text = re.sub("[^a-zA-Z]", " ", input)
    text = text.lower()
    text = text.split()
    # List comprehension
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    text = " ".join(text)
    corpus.append(text)
    one_hot_rep = [one_hot(input_text=word, n=11000) for word in corpus]
    pad = pad_sequences(sequences=one_hot_rep, maxlen=300, padding="pre")
    return pad


def test_model_lstm(input):
    sentence = predict_emotion_lstm(input)
    result = lb.inverse_transform([np.argmax(model.predict(sentence), axis=1)])[0]
    prob = np.max(model.predict(sentence))
    return result, prob


final_val = test_model_lstm("I am so surprised that he did this!!")
print("The emotion is:", final_val[0], "with a probability of:", final_val[1])
