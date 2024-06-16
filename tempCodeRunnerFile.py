import pandas as pd
from dotenv import load_dotenv
import os
import base64
from requests import post, get
import json
import random
from flask import Flask, request, render_template
import pickle
from preprocess import data_clean_lemm

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

app = Flask(__name__)

df = pd.read_csv('filtered_spotify_data.csv')

svm_model = pickle.load(open("model_data/svc_model.pkl", "rb"))
label_encoder = pickle.load(open("model_data/label_encoder.pkl", "rb"))
tfid_vectorizer = pickle.load(open("model_data/tf_vectorizer.pkl", "rb"))

emotions_dict = {
    "sadness": 0.4,
    "joy": 0.9,
    "love": 0.6,
    "anger": 0.2,
    "fear": 0.2,
    "surprise": 0.7,
}

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)

    if result.status_code != 200 or "access_token" not in json_result:
        raise Exception(f"Failed to retrieve access token: {json_result.get('error', 'No error info')}")
    
    token = json_result["access_token"]
    return token

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

def get_track_preview_url(token, track_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query = f"?q={track_name}&type=track&limit=1"
    query_url = url + query

    result = get(query_url, headers=headers)
    json_result = json.loads(result.content)
    
    if result.status_code != 200 or "tracks" not in json_result or "items" not in json_result["tracks"]:
        print(f"Failed to search for track: {json_result.get('error', 'No error info')}")
        return None
    
    items = json_result["tracks"]["items"]
    if len(items) == 0:
        print("No track with this name")
        return None

    return items[0].get('preview_url', None)

def recommend_songs_by_emotion(emotion):
    if emotion not in emotions_dict:
        return []
    
    target_valence = emotions_dict[emotion]
    closest_songs = []
    seen_songs = set()
    threshold = 0.05
    
    for idx, row in df.iterrows():
        valence = row['valence']
        song_name = row['track_name']
        
        difference = abs(valence - target_valence)
        
        if difference <= threshold and song_name not in seen_songs:
            closest_songs.append((idx, song_name, valence, difference))
            seen_songs.add(song_name)
  
    closest_songs.sort(key=lambda x: x[3])
    
    random.shuffle(closest_songs)
    
    closest_songs = closest_songs[:10]
    
    return closest_songs

def predict_emotion(text, model, vectorizer, label_encoder):
    clean_text = data_clean_lemm(text)
    input_vec = vectorizer.transform([clean_text])
    predicted_label = model.predict(input_vec)[0]
    emotion = label_encoder.inverse_transform([predicted_label])[0]
    return emotion

@app.route('/')
def index():
    return render_template('index.html', tracks=None)

@app.route('/search', methods=['POST'])
def search():
    user_input = request.form['query']
    emotion = predict_emotion(user_input, svm_model, tfid_vectorizer, label_encoder)
    
    print(f"Predicted Emotion: {emotion}")
    
    recommended_songs = recommend_songs_by_emotion(emotion)
    
    tracks = []
    if recommended_songs:
        token = get_token()
        for song_info in recommended_songs:
            preview_url = get_track_preview_url(token, song_info[1])
            if preview_url:
                tracks.append({
                    'name': song_info[1],
                    'valence': song_info[2],
                    'preview_url': preview_url
                })

    return render_template('index.html', tracks=tracks, emotion=emotion)

if __name__ == "__main__":
    app.run(debug=True)
