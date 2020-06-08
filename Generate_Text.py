# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:11:31 2020

@author: Shubham
"""

import numpy as np
import random
import pickle
from tensorflow.keras.models import model_from_json

# Deserializing model from json file
with open('model.json', 'r') as json_file:
    model_json = json_file.read()
    
model = model_from_json(model_json)
model.load_weights("model.h5")

chars, text, char_index, index_char, seq_len = pickle.load(open('Saved_Files/data.pkl', 'rb'))

def encode_seed(seed):
    x = np.zeros((1, seq_len, len(chars)), dtype = np.bool)
    for i, char in enumerate(seed):
        x[0, i, char_index[char]] = 1
    return x

def sample(preds, temperature = .5):
    preds = np.asarray(preds).astype('float64')
    with np.errstate(divide = 'ignore'):
        preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text():
    start = random.randint(0, len(text) - seq_len)
    seed = text[start: start + seq_len]
    generated = ''
    for i in range(100):
        input_ = encode_seed(seed)
        preds = model.predict(input_)[0]
        next_char = index_char[sample(preds)]
        generated += next_char
        seed = seed[1:] + next_char
    return generated

generate = generate_text()










