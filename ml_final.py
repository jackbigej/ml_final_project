import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

# map personality types to integers

personalitySequencer = { 
	'ENTP': 0,
	'ENTJ': 1,
	'ESTP': 2,
	'ESTJ': 3,
	'INTP': 4,
	'INTJ': 5,
	'ISTP': 6,
	'ISTJ': 7,
	'ENFP': 8,
	'ENFJ': 9,
	'ESFP': 10,
	'ESFJ': 11,
	'ISFP': 12,
	'ISFJ': 13,
	'INFP': 14,
	'INFJ': 15
}



def load_data():
	dataframe = pd.read_csv('mbti_1.csv', engine='python')
	dataset = dataframe.values
	
	return dataset

def make_model():

	model = Sequential()
	model.add(Dense(150, input_dim=160098, activation='relu'))
	model.add(Dense(1, activation='relu'))


	return model

def split_xy(dataset):
	dataX, dataY = [], []

	for entry in dataset:
		dataX.append(entry[1])
		dataY.append(entry[0])

	return dataX, dataY

def main():
	dataset = load_data()

	dataX, dataY = split_xy(dataset)

# tokenize texts so they can be represented numerically
	t = Tokenizer()
	t.fit_on_texts(dataX)
	dataX = t.texts_to_matrix(dataX, mode='count')

# map training target data to integers
	mappedPersonalities = []
	for mbti in dataY:
		mappedPersonalities.append(personalitySequencer[mbti])

	dataY = np.array(mappedPersonalities)

	train = int(len(dataset)*.60)
	test = len(dataset) - int(len(dataset)*.2)

	trainX, trainY = dataX[0:train], dataY[0:train]
	valX, valY = dataX[train:test], dataY[train:test]
	testX, testY = dataX[test:len(dataset) +1], dataY[test:len(dataset) +1]

	print(len(trainX))
	model = make_model()

	model.compile(loss='mean_squared_error', optimizer='adam')

	history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=10, batch_size=15, verbose=1)

	history = history.history

	results = model.evaluate(valX, valY)

	print(history)
	print(results)
	


if __name__ == '__main__':
	main()
