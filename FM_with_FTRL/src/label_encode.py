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
import datetime
import traceback
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class label_encode_processing(object):
	def __init__(self, total_frame):
		self.total_frame = total_frame

	def label_encode_trans(self):
		# initialize function
		label_encoder = LabelEncoder()

		# keep loop version
		for col in self.total_frame.columns.values:
			# encoding only categorical variables
			if self.total_frame[col].dtypes=='object':
				# label encode
				self.total_frame[col] = label_encoder.fit_transform(self.total_frame[col])

		# return new data_frame
		return self.total_frame

