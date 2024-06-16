import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import one_hot
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model

from sklearn.preprocessing import LabelEncoder
from preprocess import text_cleaning_lstm, predict_emotion_lstm

classes = ["sadness", "joy", "love", "anger", "fear", "surprise"]


train = pd.read_json("data/data.jsonl", lines=True)
train["label_name"] = train["label"].apply(lambda x: classes[x])
train = train.drop("label", axis=1)
train["length"] = [len(x) for x in train.text]

test = pd.read_json("data/test.jsonl", lines=True)
test["label_name"] = test["label"].apply(lambda x: classes[x])

lb = LabelEncoder()
train["label_name"] = lb.fit_transform(train["label_name"])
test["label_name"] = lb.fit_transform(test["label_name"])

print(train.head())
x_train_lstm = text_cleaning_lstm(train, "text", 11000, 300)
y_train_lstm = to_categorical(train["label_name"])

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
#     model.save("model_data/lstm_model_heavy.h5")
#     return model


# model = model_train(x_train_lstm, y_train_lstm)

# model = load_model("model_data/lstm_model_2.h5")


# def test_model_lstm(input):
#     sentence = predict_emotion_lstm(input)
#     result = lb.inverse_transform([np.argmax(model.predict(sentence), axis=1)])[0]
#     prob = np.max(model.predict(sentence))
#     return result, prob


# final_val = test_model_lstm("I am very sad.")
# print("The emotion is:", final_val[0], "with a probability of:", final_val[1])
