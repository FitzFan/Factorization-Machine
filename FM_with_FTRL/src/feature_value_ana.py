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
import pandas as pd
import numpy as np


"""
- 在shell的分析上，加入feature的type分析
- 脚本会在feature_value_ana.sh中被调用
"""

def value_ana(data_out, data_path):
	# 读取shell分析的数据结果
	data_value = pd.read_csv(data_out, sep='\t')

	# 看一下每个字段的取值类型
	col_dtypes = []
	sample_data = pd.read_csv(data_path, sep=',', nrows=10) # 仅读取十行
	for col in data_value['col_name']:
		col_dtypes.append(sample_data[col].dtypes)

	# 添加新的列
	data_value['col_type'] = col_dtypes
	data_value['python_num'] = data_value['col_num'] - 1

	# 重新调整列的顺序
	col_list = ['col_num','python_num','col_name','value_num','col_type']
	data_value = data_value[col_list]

	# 重新写出
	data_value.to_csv(data_out, seq='\t',header=True, index=False)

	print data_value

def main():
	# 设置路径
	data_out  = '../out_put/feature_ana.dat'
	data_path = '../data/feature_ryan.dat'

	# call function
	value_ana(data_out, data_path)

if __name__ == '__main__':
	main()