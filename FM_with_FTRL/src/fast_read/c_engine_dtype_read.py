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


data_path = sys.argv[1]
data_path = str(data_path)

print '----------------------------------------------------------------------'
print 'Set dtype as object to call read_csv function and set engine as C'

start = time.clock()
data_samples = pd.read_csv(data_path, sep=',', usecols=[2],dtype='object', error_bad_lines=False, engine='c')

ela_sec = time.clock() - start
ela_min = float(ela_sec)/60
print 'End To Read_Data, time Used:', ela_min,'min'