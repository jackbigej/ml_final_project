import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Embedding
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

def load_data():
	return pd.read_csv('mbti_1.csv', engine='python')
	
def make_basic_model(train, valid):
	model = Sequential()
	model.add(Dense(150, input_dim=10090, activation='relu'))
	model.add(Dense(1, activation='relu'))
	return model

	
def make_advanced_model(train, valid):
	model = Sequential()
	model.add(Dense(150, input_dim=10090, activation='relu'))
	model.add(Dense(1, activation='relu'))
	return model

def split_xy(dataset):
	dataX, dataY = [], []

	for entry in dataset:
		dataX.append(entry[1])
		dataY.append(entry[0])

	return dataX, dataY

dataset = load_data()

encoder = LabelEncoder()
encoder.fit(dataset['type'].values)


train=dataset.sample(frac=0.8,random_state=200)
test=dataset.drop(train.index)
  
tokenizer = Tokenizer(oov_token='<UNK>')
tokenizer.fit_on_texts(train['posts'].values)
tokenizer.num_words = 400
  
train['Y'] = encoder.transform(train['type'].values)
test['Y'] = encoder.transform(test['type'].values)

train_x, train_y = zip(*train[['posts', 'Y']].values)

test_x, test_y =zip(*test[['posts','Y']].values)
test_x = np.array(test_x)
test_y = np.array(test_y)
print(test_x)
print(test_y)

print("=======================>")
train_x = np.array(train_x)
train_y = np.array(train_y)


#train_x = tokenizer.texts_to_matrix(mode='binary')
  
train_x = tokenizer.texts_to_sequences(train_x)
train_x = pad_sequences(train_x, maxlen=5000, padding='pre', truncating='post')

model = Sequential()
# output vector size, vocab size, input sequence length
model.add(Embedding(1000, 400, input_length=5000))
model.add(LSTM(200))
model.add(Dense(200))
model.add(Dense(len(encoder.classes_), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
  
onehot = OneHotEncoder(sparse=False)
train_y = train_y.reshape((-1,1))
train_y = onehot.fit_transform(train_y)

print(train_x)
print(train_y)

model.fit(train_x, train_y, epochs=1)

# make predictions
trainPredict = model.predict(train_x)
testPredict = model.predict(test_y)
print("TestPred= ",testPredict)