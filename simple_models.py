import pandas as pd
import numpy as np
import pickle

import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from classifiers import classifiers
from preprocess import data_clean_stem, data_clean_lemm


classes = ["sadness", "joy", "love", "anger", "fear", "surprise"]


train = pd.read_json("data/transform/train.jsonl", lines=True)
train["label_name"] = train["label"].apply(lambda x: classes[x])
train = train.drop("label", axis=1)
train["length"] = [len(x) for x in train.text]

test = pd.read_json("data/transform/test.jsonl", lines=True)
test["label_name"] = test["label"].apply(lambda x: classes[x])


# Preprocessing
# lb encodes the labels for the sentiments numerically
lb = LabelEncoder()
train["label_name"] = lb.fit_transform(train["label_name"])
test["label_name"] = lb.fit_transform(test["label_name"])
encoded_df = train.copy()


tqdm.pandas()


encoded_df["cleaned_stem"] = encoded_df["text"].progress_apply(data_clean_stem)
encoded_df["cleaned"] = encoded_df["text"].progress_apply(data_clean_lemm)


X_train, X_test, Y_train, Y_test = train_test_split(
    encoded_df["cleaned_stem"], encoded_df["label_name"], test_size=0.2, random_state=42
)

# Feature extraction
tfidvec = TfidfVectorizer()
X_train_tf = tfidvec.fit_transform(X_train)
X_test_tf = tfidvec.transform(X_test)


lg, lg_y_predict, svc, svc_y_predict = classifiers(
    X_train_tf, X_test_tf, Y_train, Y_test
)

lg_cm = confusion_matrix(Y_test, lg_y_predict)
svm_cm = confusion_matrix(Y_test, svc_y_predict)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(confusion_matrix=lg_cm).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("Logistic Regression Confusion Matrix")

ConfusionMatrixDisplay(confusion_matrix=svm_cm).plot(ax=axes[1], cmap="Blues")
axes[1].set_title("SVM Confusion Matrix")

plt.show()


def predict_emotion(input, model):
    clean = data_clean_lemm(input)
    input_vec = tfidvec.transform([clean])
    predicted_label = model.predict(input_vec)[0]
    emotion = lb.inverse_transform([predicted_label])[0]
    label = np.max(lg.predict(input_vec))
    return emotion, label


# encoded_test_df = test.copy()

# encoded_test_df["true_label"] = lb.inverse_transform(encoded_test_df["label_name"])

# correct_predicted = 0


# for i in range(len(encoded_test_df)):
#     emotion_lg, label_lg = predict_emotion(encoded_test_df["text"][i], lg)

#     if emotion_lg == encoded_test_df["true_label"][i]:
#         correct_predicted += 1

# print(correct_predicted / len(encoded_test_df) * 100)


# folder = "model_data"

# if not os.path.exists(folder):
#     os.makedirs(folder)

# if not os.listdir(folder):
#     pickle.dump(lg, open("model_data/lg_model.pkl", "wb"))
#     pickle.dump(svc, open("model_data/svc_model.pkl", "wb"))
#     pickle.dump(lb, open("model_data/label_encoder.pkl", "wb"))
#     pickle.dump(tfidvec, open("model_data/tf_vectorizer.pkl", "wb"))
