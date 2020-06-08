# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:09:20 2020

@author: Shubham
"""

import numpy as np
import re
import pickle
import requests
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import model_from_json

response = requests.get('https://www.linguistik.uzh.ch/dam/jcr:169bff5c-ac13-457b-9acb-4fe7f1ad5cb0/Harry%20Potter%20and%20the%20Sorcerer.txt')
text = response.text
text = re.sub('[^\x00-\x7f]', '', text)
text = text[80:].lower()

chars = sorted(list(set(text.lower())))
char_index = {char: i for i, char in enumerate(chars)}
index_char = [char for char in chars]
vocab_size = len(chars)
seq_len = 30
steps = 1
batches = 128
epochs = 10

pickle.dump((chars, text, char_index, index_char, seq_len), open('Saved_Files/data.pkl', 'wb'))

def get_sequences(text, seq_len, steps):
    sequences = []
    next_chars = []
    for i in range(0, len(text) - seq_len, steps):
        sequences.append(text[i: i + seq_len])
        next_chars.append(text[i + seq_len])
    return sequences, next_chars

def encode_chars(sequences, next_chars, chars):
    layers = len(sequences)
    rows = len(sequences[0])
    columns = len(chars)
    x = np.zeros((layers, rows, columns), dtype = np.bool)
    y = np.zeros((layers, columns), dtype = np.bool)
    for i, sequence in enumerate(sequences):
        for j, char in enumerate(sequence):
            x[i, j, char_index[char]] = 1
        y[i, char_index[next_chars[i]]] = 1
    return x, y

def build_model(seq_len, vocab_size):
    model = Sequential()
    model.add(CuDNNLSTM(units = 128, return_sequences = True, input_shape = (seq_len, vocab_size)))
    model.add(Dropout(0.2))
    model.add(CuDNNLSTM(units = 128))
    model.add(Dropout(0.2))
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units = vocab_size, activation = 'softmax'))
    return model

def train_model(model, x, y, batches, epochs):
    model.compile(optimizer = RMSprop(lr = 0.001), loss = 'categorical_crossentropy')
    model.fit(x, y, batch_size = batches, epochs = epochs)
    
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')
    
    return model

sequences, next_chars = get_sequences(text, seq_len, steps)
x, y = encode_chars(sequences, next_chars, chars)

model = build_model(seq_len, vocab_size)
model = train_model(model, x, y, batches, epochs)










