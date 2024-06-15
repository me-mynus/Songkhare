# import numpy as np

# from keras.models import Sequential
# from keras.layers import Embedding, LSTM, Dense, Dropout
# from keras.preprocessing.text import one_hot
# from keras.callbacks import EarlyStopping
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.models import load_model

# from preprocess import text_cleaning_lstm

# x_train_lstm = text_cleaning_lstm(train_df, "text", 11000, 300)
# y_train_lstm = to_categorical(train_df["sentiment"])


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

# model = load_model("model_data/lstm_model_2.h5")


# def predict_emotion_lstm(input):
#     stemmer = PorterStemmer()
#     corpus = []
#     text = re.sub("[^a-zA-Z]", " ", input)
#     text = text.lower()
#     text = text.split()
#     # List comprehension
#     text = [stemmer.stem(word) for word in text if word not in stopwords]
#     text = " ".join(text)
#     corpus.append(text)
#     one_hot_rep = [one_hot(input_text=word, n=11000) for word in corpus]
#     pad = pad_sequences(sequences=one_hot_rep, maxlen=300, padding="pre")
#     return pad


# def test_model_lstm(input):
#     sentence = predict_emotion_lstm(input)
#     result = lb.inverse_transform([np.argmax(model.predict(sentence), axis=1)])[0]
#     prob = np.max(model.predict(sentence))
#     return result, prob


# final_val = test_model_lstm("I am very sad.")
# print("The emotion is:", final_val[0], "with a probability of:", final_val[1])
