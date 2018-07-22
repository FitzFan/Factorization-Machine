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
import warnings
import commands
import pandas as pd 
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from FM_FTRL import FM
from LR_FTRL import LR, evaluate_model, get_auc
from preprocessing import data_preprocess

"""
- python implementation of Factorization Machines and Logistic Regression
- optimizer is FTRL
"""

class FTRL:
    def __init__(self, base, args_parse, iteration):
        """
        initialize the ftrl model
        :param base: the base model
        :param args_parse: the according parameters
        :param iteration: the stopping criterion
        """
        self.iteration = iteration
        self.base = base
        self.params = {}
        if self.base == "lr":
            dim = args_parse['dim']
            alpha = args_parse['alpha_w']
            beta = args_parse['beta_w']
            lambda1 = args_parse['lambda_w1']
            lambda2 = args_parse['lambda_w2']
            self.model = LR(dim=dim, alpha=alpha, beta=beta, lambda1=lambda1, lambda2=lambda2)
        else:
            dim = args_parse['dim']
            dim_map = args_parse['dim_map']
            sigma = args_parse['sigma']
            alpha_w = args_parse['alpha_w']
            alpha_v = args_parse['alpha_v']
            beta_w = args_parse['beta_w']
            beta_v = args_parse['beta_v']
            lambda_w1 = args_parse['lambda_w1']
            lambda_w2 = args_parse['lambda_w2']
            lambda_v1 = args_parse['lambda_v1']
            lambda_v2 = args_parse['lambda_v2']
            self.model = FM(dim=dim, dim_lat=dim_map, sigma=sigma,
                            alpha_w=alpha_w, alpha_v=alpha_v, beta_w=beta_w, beta_v=beta_v,
                            lambda_w1=lambda_w1, lambda_w2=lambda_w2, lambda_v1=lambda_v1, lambda_v2=lambda_v2)

    def fit(self, train_samples, train_labels):
        """
        train model using ftrl optimization model
        :param train_samples:  the training samples         -shapes(n_samples, dimension)
        :param train_labels:   the training labels          -shapes(n_samples, )
        """
        self.model.train_ftrl(train_samples, train_labels, self.iteration, is_print=True)
        self.params['weights'] = self.model.weights
        self.params['bias'] = self.model.weights[-1]
        if self.base == "fm":
            self.params['V'] = self.model.V

        return self.params

    def predict(self, test_samples):
        """
        test the unseen samples using the trained model
        :param test_samples: the testing samples            -shapes(n_samples, dimension)
        :return: the predictions
        """
        test_preds = self.model.predict(test_samples)
        return test_preds

    def evaluate(self, test_samples, test_labels, metrics='error'):
        """
        evaluate the model using different metrics
        :param test_samples: the testing samples            -shapes(n_samples, dimension)
        :param test_labels: the testing labels              -shapes(n_samples,)
        :param metrics: auc or error
        :return: the evaluation
        """
        test_preds = self.predict(test_samples)
        if metrics == 'error':
            evaluation = evaluate_model(preds=test_preds, labels=test_labels)
        else:
            evaluation = roc_auc_score(y_true=test_labels, y_score=test_preds)
        return evaluation

def read_data(data_path, tar_col=None):
    if tar_col is not None:
        data_samples = pd.read_csv(data_path, sep=',', usecols=tar_col)
    else:
        data_samples = pd.read_csv(data_path, sep=',')

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    return data_samples

def preprocess(raw_data_path, need_label_encode_dic, need_one_hot_dic):
    # time clock
    start = time.clock()

    print 'Begin To label encode'

    # label-encode processing
    for col_name in need_label_encode_dic:
        # read data 
        data_samples = read_data(raw_data_path, [need_label_encode_dic[col_name]])

        # 设一个保险
        if data_samples.columns[0] != col_name:
            print 'maybe index is wrong, please debug...'
            print 'data_samples.columns is', data_samples.columns[0]
            print 'col_name is', col_name

        # call function
        data_preprocess_method = data_preprocess(data_samples, col_name)
        data_preprocess_method.run()

        print col_name, 'done to label encode'

    print '\nBegin To one hot encode'

    # gabage collect
    del data_samples
    gc.collect()

    # one-hot processing
    for col_name in need_one_hot_dic:
        # read data 
        data_samples = read_data(raw_data_path, [need_one_hot_dic[col_name]])

        # 设一个保险
        if data_samples.columns[0] != col_name:
            print 'maybe index is wrong, please debug...'
            print 'data_samples.columns is', data_samples.columns[0]
            print 'col_name is', col_name

        # call function
        data_preprocess_method = data_preprocess(data_samples, col_name, 'one_hot')
        data_preprocess_method.run()

        print col_name, 'done to one hot'
    
    # gabage collect
    del data_samples
    gc.collect()

    # 合并所有的中间数据
    paste_cmd = 'paste -d ',' prepro_*.dat > ../data/all_featrue_after_preprocessing.dat'
    paste_out = commands.getstatusoutput(paste_cmd)

    if str(paste_out[0]) == '0':
        print 'Succeed to paste all prepro_*.dat to all_featrue_preprocessing.dat'

        # check rows of all_featrue_preprocessing.dat
        wc_cmd = 'wc -l all_featrue_after_preprocessing.dat'
        wc_out = commands.getstatusoutput(wc_cmd)

        if str(wc_out[0]) == '0':
            print 'rows of all_featrue_preprocessing.dat is',wc_out[1][0:wc_out[1].find(' ')]
        else:
            print wc_out[1]
    else:
        print paste_out[1]

    print 'End To data preprocessing, time Used:', (time.clock() - start)
    print '+------------------------------------------+' 
    
def train_model(X_train, X_test, y_train, y_test, hyper_params, iteration_):  
    # time clock
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print '+------------------------------------------+'
    print 'Begin To Train Model:', start_time
    
    # create models
    lr = FTRL(base="lr", args_parse=hyper_params, iteration=iteration_)
    fm = FTRL(base="fm", args_parse=hyper_params, iteration=iteration_)

    # train models
    params = fm.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # test the unseen samples
    test_preds_lr = lr.predict(X_test)
    test_preds_fm = fm.predict(X_test)
    test_error_lr = evaluate_model(test_preds_lr, y_test)
    test_error_fm = evaluate_model(test_preds_fm, y_test)
    test_auc_lr = roc_auc_score(y_true=y_test, y_score=test_preds_lr)
    test_auc_fm = roc_auc_score(y_true=y_test, y_score=test_preds_fm)
    my_test_auc_lr = get_auc(scores=test_preds_lr, labels=y_test)
    my_test_auc_fm = get_auc(scores=test_preds_fm, labels=y_test)

    print("logistic regression-test error: %.2f%%" % test_error_lr)
    print("logistic regression-test auc: ", test_auc_lr)
    print("logistic regression-my test auc: ", my_test_auc_lr)
    print '+------------------------------------------+'

    print("factorization machine-test error: %.2f%%" % test_error_fm)
    print("factorization machine-test auc: ", test_auc_fm)
    print("factorization machine-my test auc: ", my_test_auc_fm)
    print '+------------------------------------------+'

    # test the unseen samples
    test_preds = fm.predict(X_test)
    test_error = evaluate_model(test_preds, y_test)
    test_auc = roc_auc_score(y_true=y_test, y_score=test_preds)
    my_auc = get_auc(scores=test_preds, labels=y_test)
    print("test-error: %.2f%%" % test_error)
    print("test-sklearn auc: ", test_auc)
    print("test-my auc: ", my_auc)
    print '+------------------------------------------+'

    # print the parameters of trained FM model
    print("weights: ", params['weights'])
    print("bias: ", params['bias'])
    print("V: ", params['V']) 
    print '+------------------------------------------+'

    # time clock
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    print 'End To Train Model:', end_time
    print '+------------------------------------------+'

def main():
    # ignore warnings
    warnings.filterwarnings("ignore")

    # set path of raw data
    raw_data_path = '../data/feature_ryan.dat'

    # set features needed to preprocessing
    """
    - dict:
        - key：col_name
        - value:col_num
    - one_hot 和 label_encode只需要做其中一个
    - 坑：shell索引的起点是1，python是0。
    """
    need_label_encode_dic = {'site_id':2,
                            'site_domain':3,
                            'app_id':5,
                            'device_id':8,
                            'device_ip':9,
                            'device_model':10,
                           }

    need_one_hot_dic = {'site_category':4,
                        'app_domain':6,
                        'app_category':7,
                        'device_type':11,
                        'device_conn_type':12
                        }

    # data preprocessing
    """
    增加更大数据量的压力测试
    """
    preprocess(raw_data_path, need_label_encode_dic, need_one_hot_dic)

    # set path of preprocessed data
    data_samples = read_data('../data/all_featrue_after_preprocessing.dat')
    target_samples = read_data('../data/label_ryan_dat')

    # data_set basic info
    print '+----------------------+'
    print 'feature dim is', len(data_samples.columns)
    print 'total  rows is', len(data_samples)
    print 'size of object is', format(sys.getsizeof(data_samples)/1024.0/1024.0, '0.2f'), 'MB'
    print '+----------------------+'

    # split all the samples into training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(data_samples, target_samples, test_size=0.2, random_state=24)
 
    # gabage collect
    del data_samples, target_samples
    gc.collect()

    # define hyper_params
    hyper_params = {
        'dim': dim_,
        'dim_map': 8,
        'sigma': 1.0,
        'alpha_w': 0.2,
        'alpha_v': 0.2,
        'beta_w': 0.2,
        'beta_v': 0.2,
        'lambda_w1': 0.2,
        'lambda_w2': 0.2,
        'lambda_v1': 0.2,
        'lambda_v2': 0.2,
        }
    iteration_ = 100

    # convert dataFrame to ndarray
    X_train = X_train.values
    X_test  = X_test.values
    y_train = y_train.values.flatten()  # shape must be like (10000,)
    y_test  = y_test.values.flatten()

    print '+------------------------------------------+'
    print "type of X_train is", type(X_train)
    print "type of X_test  is", type(X_test)
    print "type of y_train is", type(y_train)
    print "type of y_test  is", type(y_test)

    # train model
    train_model(X_train, X_test, y_train, y_test, hyper_params, iteration_)

    """
    还差：save model and test use model
    """

if __name__ == "__main__":
    main()

