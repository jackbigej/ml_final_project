import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

def load_data():
	dataframe = pd.read_csv('mbti_1.csv', engine='python')
	dataset = dataframe.values
	
	return dataset

def make_model(train, valid):

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

def main():
	dataset = load_data()

	'''	
	m = 0

	for i in dataset:
		if len(i[1]) > m:
			m = len(i[1])



	print(m)
	
	'''

	train = int(len(dataset)*.60)
	test = len(dataset) - int(len(dataset)*.2)

	train, valid, test = dataset[0:train, :], dataset[train:test], dataset[test:len(dataset) + 1]

	print('train: ', len(train), ', valid: ', len(valid), ', test: ', len(test))

	#print(train[1])
	
	for i in train:
		while len(i[1]) < 10090:
			i[1] += ' '
	'''
	for i in dataset:
		print(len(i[1]))
	
	'''

	model = make_model(train, valid)

	model.compile(loss='mean_squared_error', optimizer='adam')

	trainx, trainy = split_xy(train)

	print(len(trainx))

	valx, valy = split_xy(valid)

	print(len(trainy))

	history = model.fit(trainx, trainy, validation_data=(valx, valy), epochs=10, batch_size=15, verbose=1)

	history = history.history

	results = model.evaluate(valx, valy)

	print(history)
	print(results)
	


if __name__ == '__main__':
	main()
