#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : Ryan Fan 
@E-Mail  : ryanfan0528@gmail.com
@Version : v1.0
"""

import os
import time
import sys
import re
import gc
import datetime
import traceback
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class one_hot_processing(object):
	def __init__(self, train_frame, test_frame, one_hot_cols=None):
		self.train_frame = train_frame
		self.test_frame = test_frame
		if one_hot_cols is None:
			self.one_hot_cols = self.train_frame.columns.values
		else:
			self.one_hot_cols = one_hot_cols

	def label_encode_trans(self):
		# initialize function
		label_encoder = LabelEncoder()

		for col in self.train_frame.columns.values:
			# using whole data to form an exhaustive list of levels
			data = self.train_frame[col].append(self.test_frame[col])

			# label encode
			self.train_frame[col] = label_encoder.fit_transform(self.train_frame[col])
			self.test_frame[col] = label_encoder.fit_transform(self.test_frame[col])

	def one_hot_trans(self):
		# initialize function
		one_hot_encoder = OneHotEncoder()

		for col in self.one_hot_cols:
			# creating an exhaustive list of all possible categorical values
			data = self.train_frame[[col]].append(self.test_frame[[col]])

			# Fitting One Hot Encoding on train data and test data
			one_hot_encoder.fit(data)

			# garbage collect
			del data
			gc.collect()

			# transform to array after one_encoding
			temp_train = one_hot_encoder.fit_transform(self.train_frame[[col]].values.reshape(-1,1)).toarray()
			temp_test = one_hot_encoder.fit_transform(self.test_frame[[col]].values.reshape(-1,1)).toarray()

			# changing the encoded features into a data frame with new column names
			temp_train = pd.DataFrame(temp_train, columns = [col + '_' + str(int(i)) for i in range(temp_train.shape[1])])
			temp_test = pd.DataFrame(temp_test, columns = [col + '_' + str(int(i)) for i in range(temp_test.shape[1])])

			# setting the index values similar to the self.train_frame data frame
			temp_train = temp_train.set_index(self.train_frame.index.values)
			temp_test = temp_test.set_index(self.test_frame.index.values)

			# adding the new One Hot Encoded varibales to the train data frame
			self.train_frame = pd.concat([self.train_frame, temp_train], axis=1)
			self.test_frame = pd.concat([self.test_frame, temp_test], axis=1)

			# garbage collect
			del temp_train, temp_test
			gc.collect()

			# del old column
			self.train_frame.drop([col], axis=1, inplace=True)
			self.test_frame.drop([col], axis=1, inplace=True)

		# garbage collect
		del temp_train, temp_test
		gc.collect()

	def run(self):
		# encode all the categorical features.
		self.label_encode_trans()

		# encode the data by one-hot method
		self.one_hot_trans()

		# return new data_frame
		return self.train_frame, self.test_frame

def main():
	# give a demo as following
	path='../data/'
	train_frame = pd.read_csv(path+'X_train.csv')
	test_frame = pd.read_csv(path+'X_test.csv')

	# features need to one-hot
	need_one_hot_features = ['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Credit_History', 'Property_Area']

	# call one_hot_processing class
	one_hot_method = one_hot_processing(train_frame, test_frame, need_one_hot_features)
	one_hot_train,one_hot_test = one_hot_method.run()

	# print results of demo
	print one_hot_train[['Dependents_0','Dependents_1','Dependents_2','Dependents_3']].head(10)

# if __name__ == '__main__':
# 	main()