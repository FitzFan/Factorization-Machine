#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : Ryan Fan 
@E-Mail  : ryanfan0528@gmail.com
@Version : v1.0
"""

import os
import re
import gc
import time
import sys
import csv
import datetime
import warnings
import numpy as np
import pandas as pd 

from label_encode import label_encode_processing
from one_hot import one_hot_processing


"""
data_preprocess功能：
	- 对传入的特征进行label_encode
	- 对需要one-hot的特征进行one-hot
	- 结果写出到中间数据文件夹
"""

class data_preprocess(object):
	def __init__(self, total_frame, col_name, task='encode'):
		self.total_frame = total_frame
		self.col_name = col_name
		self.task = task

	def label_encode(self):
		label_encode_method = label_encode_processing(self.total_frame) # 默认是对所有的object类型的字段进行转换
		self.total_frame = label_encode_method.label_encode_trans()

	def one_hot_encode(self):
		# get dimensions of dataSet
		num_samples, dim_ = self.total_frame.shape
		print 'Before One Hot Encoding', 'dimension of', self.col_name, 'is', dim_
		
		# one-hot processing
		one_hot_method = one_hot_processing(self.total_frame, self.col_name)
		self.total_frame = one_hot_method.one_hot_trans() # 返回的是dataFrame

		# one-hot 效果验证
		num_samples, dim_ = self.total_frame.shape

		print 'After One Hot Encoding', 'dimension of', self.col_name, 'is', dim_
		print '+------------------------------------------+'

	def run(self):
		# always need to label encode
		self.label_encode()

		# 判断是否需要做one-hot
		if self.task == 'one_hot':
			# call function
			self.one_hot_encode()

			# 设置文件名
			file_name = self.total_frame.columns[0]
			file_name = file_name[0:file_name.find('_#$#_')]
		else:
			# 设置文件名
			file_name = self.total_frame.columns[0]

		# 创建中间数据存储目录
		temp_data_path='../temp_data/'
		if not os.path.isdir(temp_data_path):
			os.mkdir(temp_data_path)

		# 写出文件
		self.total_frame.to_csv(temp_data_path+'prepro_'+file_name+'.dat', sep=',', header=True, index=False)

		print 'done to write out prepro_'+file_name+'.dat'
		print '+------------------------------------------+'


