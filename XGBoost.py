import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming you have a dataset named 'data.csv' with columns 'emotions' and 'text'
# Load the dataset
data = pd.read_csv('ISEAR_dataset.csv')

# Text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join the tokens back into a single string
    text = ' '.join(filtered_tokens)
    return text

# Apply text preprocessing to the 'text' column
data['processed_text'] = data['content'].apply(preprocess_text)

# Encode emotions to numerical labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['sentiment'])

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train = tfidf_vectorizer.fit_transform(train_data['processed_text'])
y_train = train_data['label']
X_test = tfidf_vectorizer.transform(test_data['processed_text'])
y_test = test_data['label']

# Train XGBoost classifier
model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_))
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
