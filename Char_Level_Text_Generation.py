# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:09:20 2020

@author: Shubham
"""

# Import libraries

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

# Read data (Harry Potter and the Sorcerer's Stone)

response = requests.get('https://www.linguistik.uzh.ch/dam/jcr:169bff5c-ac13-457b-9acb-4fe7f1ad5cb0/Harry%20Potter%20and%20the%20Sorcerer.txt')
text = response.text
text = re.sub('[^\x00-\x7f]', '', text)
text = text[80:].lower()

# Create required parameters

chars = sorted(list(set(text.lower())))
char_index = {char: i for i, char in enumerate(chars)}
index_char = [char for char in chars]
vocab_size = len(chars)
seq_len = 30
steps = 1
batches = 128
epochs = 10

# dump parameters in data.pkl. Will be used while generating texts

pickle.dump((chars, text, char_index, index_char, seq_len), open('Saved_Files/data.pkl', 'wb'))

# Create list of lists. Each list in the list is sequnce of characters stored in 'sequences' (independet variable) and 
# respective next character stored in 'next_char' (dependent variable).

def get_sequences(text, seq_len, steps):
    sequences = []
    next_chars = []
    for i in range(0, len(text) - seq_len, steps):
        sequences.append(text[i: i + seq_len])
        next_chars.append(text[i + seq_len])
    return sequences, next_chars

# Encode 'sequences' and 'next_chars' into One Hot Encoding
# layers - number of sequences, rows - number of characters in each sequence, columns - number of characters in vocabulary
# x is a 3 dimentional matrix
# y is a 2 dimentional list of vectors.

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

# Create LSTM neural netwok with two LSTM layers and two Dense layers
# CuDNNLSTM (Cuda enablesd LSTM, to increase computational speed) 

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

# Compile and Train model 

def train_model(model, x, y, batches, epochs):
    model.compile(optimizer = RMSprop(lr = 0.001), loss = 'categorical_crossentropy')
    model.fit(x, y, batch_size = batches, epochs = epochs)
    # save trained model for later use
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')
    
    return model

# Method calls
sequences, next_chars = get_sequences(text, seq_len, steps)
x, y = encode_chars(sequences, next_chars, chars)
model = build_model(seq_len, vocab_size)
model = train_model(model, x, y, batches, epochs)