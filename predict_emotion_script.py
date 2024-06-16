import pickle
from preprocess import data_clean_lemm
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load the models and other objects
lg = pickle.load(open("model_data/lg_model.pkl", "rb"))
# svc = pickle.load(open("model_data/svc_model.pkl", "rb"))
lb = pickle.load(open("model_data/label_encoder.pkl", "rb"))
tfidvec = pickle.load(open("model_data/tf_vectorizer.pkl", "rb"))


def predict_emotion(input_text, model):
    clean = data_clean_lemm(input_text)
    input_vec = tfidvec.transform([clean])
    predicted_label = model.predict(input_vec)[0]
    emotion = lb.inverse_transform([predicted_label])[0]
    return emotion


if __name__ == "__main__":
    input_text = input("Enter the text to analyze: ")
    emotion = predict_emotion(input_text, lg)
    print(f"Predicted emotion: {emotion}")
