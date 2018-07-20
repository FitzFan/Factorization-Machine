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
from label_encode import label_encode_processing
from sklearn.preprocessing import OneHotEncoder


class one_hot_processing(object):
	def __init__(self, total_frame, col_name):
		self.total_frame = total_frame
		self.col = col_name

	def gabage_collect(self, val):
		del val
		gc.collect()

	def one_hot_trans(self):
		# initialize function
		one_hot_encoder = OneHotEncoder()

		# creating an exhaustive list of all possible categorical values
		data = self.total_frame[[self.col]]

		# Fitting One Hot Encoding on train data and test data
		one_hot_encoder.fit(data)

		# garbage collect
		self.gabage_collect(data)

		# transform to array after one_encoding
		temp_train = one_hot_encoder.fit_transform(self.total_frame[[self.col]].values.reshape(-1,1)).toarray()
		
		# changing the encoded features into a data frame with new column names
		temp_train = pd.DataFrame(temp_train, columns = [self.col + '_#$#_' + str(int(i)) for i in range(temp_train.shape[1])])
		
		# setting the index values similar to the self.total_frame data frame
		temp_train = temp_train.set_index(self.total_frame.index.values)
		
		# adding the new One Hot Encoded varibales to the train data frame
		self.total_frame = pd.concat([self.total_frame, temp_train], axis=1)
		
		# garbage collect
		self.gabage_collect(temp_train)

		# del old column
		self.total_frame.drop([self.col], axis=1, inplace=True)

		# return new data_frame
		return self.total_frame
		
