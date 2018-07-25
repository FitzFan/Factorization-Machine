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
import sys
import time
import datetime
import warnings
import commands
import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from FM_FTRL import FM
from LR_FTRL import LR, evaluate_model, get_auc
from preprocessing import data_preprocess
from send_email_src import send_email


"""
- DIY python implementation of Factorization Machines and Logistic Regression;
- optimizer is FTRL;
- huge_data_set one-hot algorithm;
- Mail Notifier; 
"""

"""
待尝试优化：
- read_csv()时，指定dtype。 【underdoing】
- 将object转化为category。  【to_do】
    - converted_obj.loc[:,col] = gl_obj[col].astype('category')
- 根据int类型的取值范围设置对应的int类型：【to_do】
    - uint8: [0, 255]
    - int8: [-128, 127]
    - int16:[-32768, 32767]
- pd.to_numeric() 来对数值型进行向下类型转换。【to_do】
    - 对数值型进行向下类型转换, method: apply(pd.to_numeric,downcast='unsigned')
    - float64 转换为 float32, method: apply(pd.to_numeric,downcast='float')

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

def ignore_warning():
    warnings.filterwarnings('ignore')

def read_data(data_path, tar_col=None):
    # 设置error_bad_lines=False，忽略某些异常的row
    # 读取特定列 <=> 做label_encode or one_hot <=> data_type is object
    if tar_col is not None:
        data_samples = pd.read_csv(data_path, sep=',', usecols=tar_col, error_bad_lines=False, dtype='object', engine='c')
    else:
        data_samples = pd.read_csv(data_path, sep=',', error_bad_lines=False, engine='c')

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    return data_samples

def preprocess(raw_data_path, need_label_encode_dic, need_one_hot_dic):
    # 输出邮件的内容: what to do and time used
    what_to_do_list = []
    time_used_list = []

    # time clock
    start = time.clock()

    print 'Begin To label encode'

    # label-encode processing
    for col_name in need_label_encode_dic:
        # 重启clock
        start_encode = time.clock()

        # read data 
        data_samples = read_data(raw_data_path, [need_label_encode_dic[col_name]])

        # 设一个保险
        if data_samples.columns[0] != col_name:
            print 'maybe index is wrong, please debug...'
            print 'data_samples.columns is', data_samples.columns[0]
            print 'col_name is', col_name
            sys.exit()

        # call function
        data_preprocess_method = data_preprocess(data_samples, col_name)
        data_preprocess_method.run()

        # gabage collect
        del data_samples
        gc.collect()

        # 计算时间统计
        encode_ela = time.clock() - start_encode
        encode_ela = float(encode_ela) / 60
        encode_ela = round(encode_ela, 2)

        # 发邮件的内容
        what_to_do_list.append('label_encode: ' + col_name)
        time_used_list.append(str(encode_ela) + ' min')

    print 'Begin To one hot encode'

    # one-hot processing
    for col_name in need_one_hot_dic:
        # 重启clock
        start_hot = time.clock()

        # read data 
        data_samples = read_data(raw_data_path, [need_one_hot_dic[col_name]])

        # 设一个保险
        if data_samples.columns[0] != col_name:
            print 'maybe index is wrong, please debug...'
            print 'data_samples.columns is', data_samples.columns[0]
            print 'col_name is', col_name
            sys.exit()

        # call function
        data_preprocess_method = data_preprocess(data_samples, col_name, 'one_hot')
        data_preprocess_method.run()
    
        # gabage collect
        del data_samples
        gc.collect()

        # 计算时间统计
        hot_ela = time.clock() - start_hot
        hot_ela = float(hot_ela) / 60
        hot_ela = round(hot_ela, 2)

        # 发邮件的内容
        what_to_do_list.append('one_hot: ' + col_name)
        time_used_list.append(str(hot_ela) + ' min')

    # 合并所有的中间数据（按列合并）
    paste_cmd = "paste -d ',' ../temp_data/prepro_*.dat > ../data/all_featrue_after_preprocessing.dat"
    paste_out = commands.getstatusoutput(paste_cmd)

    if str(paste_out[0]) == '0':
        print 'Succeed to paste all prepro_*.dat to all_featrue_preprocessing.dat'

        # check rows of all_featrue_preprocessing.dat
        wc_cmd = 'wc -l ../data/all_featrue_after_preprocessing.dat'
        wc_out = commands.getstatusoutput(wc_cmd)

        if str(wc_out[0]) == '0':
            print 'rows of all_featrue_preprocessing.dat is',wc_out[1][0:wc_out[1].find(' ')]
        else:
            print wc_out[1]
    else:
        print paste_out[1]

    # 计算整个data precessing的耗时
    data_pre_ela = time.clock() - start
    data_pre_ela = float(data_pre_ela) / 60
    data_pre_ela = round(data_pre_ela, 2)
    print 'End To data preprocessing, time Used: %s min' %(str(data_pre_ela))
    print '+------------------------------------------+' 

    # 添加汇总时间
    what_to_do_list.append('All Data Preprocessing Job')
    time_used_list.append(str(data_pre_ela) + ' min')

    # 设置邮件发送基本信息
    receivers = ['ryanfan0313@163.com']
    Subject = 'Data Preprocessing Report'
    table_name = 'Data Path Is "%s"'%(raw_data_path)
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 

    # 设置发邮件的内容
    all_final_top = pd.DataFrame({'Task List':what_to_do_list,
                                  'Time Used':time_used_list
                                })

    # 转换类型
    for column in all_final_top:
        all_final_top[column] = all_final_top[column].astype(str)

    # 发送邮件
    send_email_func = send_email(receivers, all_final_top, Subject, table_name, date)
    send_email_func.run()
    
def train_model(X_train, X_test, y_train, y_test, hyper_params, iteration_): 
    # time clock
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 重启clock
    start_train = time.clock()

    # 新建数据盒子
    what_to_do_list = []
    have_done_list = []

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
    what_to_do_list.append("logistic regression-test error")
    what_to_do_list.append("logistic regression-test auc")
    what_to_do_list.append("logistic regression-my test auc")
    have_done_list.append(str(test_error_lr))
    have_done_list.append(str(test_auc_lr))
    have_done_list.append(str(my_test_auc_lr))
    print '+------------------------------------------+'

    print("factorization machine-test error: %.2f%%" % test_error_fm)
    print("factorization machine-test auc: ", test_auc_fm)
    print("factorization machine-my test auc: ", my_test_auc_fm)
    what_to_do_list.append("factorization machine-test error")
    what_to_do_list.append("factorization machine-test auc")
    what_to_do_list.append("factorization machine-my test")
    have_done_list.append(str(test_error_fm))
    have_done_list.append(str(test_auc_fm))
    have_done_list.append(str(my_test_auc_fm))
    print '+------------------------------------------+'

    # test the unseen samples
    test_preds = fm.predict(X_test)
    test_error = evaluate_model(test_preds, y_test)
    test_auc = roc_auc_score(y_true=y_test, y_score=test_preds)
    my_auc = get_auc(scores=test_preds, labels=y_test)
    print("test-error: %.2f%%" % test_error)
    print("test-sklearn auc: ", test_auc)
    print("test-my auc: ", my_auc)
    what_to_do_list.append("test-error")
    what_to_do_list.append("test-sklearn auc")
    what_to_do_list.append("test-my auc")
    have_done_list.append(str(test_error))
    have_done_list.append(str(test_auc))
    have_done_list.append(str(my_auc))
    print '+------------------------------------------+'

    # print the parameters of trained FM model
    print("weights: ", params['weights'])
    print("bias: ", params['bias'])
    print("V: ", params['V']) 
    print '+------------------------------------------+'

    # time clock
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 计算训练耗时
    model_train_ela = time.clock() - start_train
    model_train_ela = float(model_train_ela) / 60
    model_train_ela = round(model_train_ela, 2)

    print 'End To Train Model:', end_time
    print '+------------------------------------------+'

    # 添加时间信息
    what_to_do_list.append('Total Train Job Time Used')
    time_used_list.append(str(model_train_ela) + ' min')

    # 设置邮件发送基本信息
    receivers = ['ryanfan0313@163.com']
    Subject = 'Model Trainning Report'
    table_name = 'data set size is %d' %(len(X_train))
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 

    # 设置发邮件的内容
    all_final_top = pd.DataFrame({'Evaluation Metrics':what_to_do_list,
                                  'Model Value':have_done_list
                                })

    # 转换类型
    for column in all_final_top:
        all_final_top[column] = all_final_top[column].astype(str)

    # 发送邮件
    send_email_func = send_email(receivers, all_final_top, Subject, table_name, date)
    send_email_func.run()

def main():
    # ignore warnings
    ignore_warning()

    # set path of data needed to use
    raw_data_path = '../data/feature_ryan.dat'
    label_data_path = '../data/label_ryan.dat'
    after_pre_data_path = '../data/all_featrue_after_preprocessing.dat'
    
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
                            'app_domain':6,
                            'device_id':8,
                            'device_ip':9,
                            'device_model':10,
                           }

    need_one_hot_dic = {'site_category':4,
                        'C1':0,
                        'C15':14,
                        'C16':15,
                        'C18':17,
                        'banner_pos':1,
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
    data_samples = read_data(after_pre_data_path) 
    target_samples = read_data(label_data_path)

    # 获取数据集基本情况
    num_samples, dim_ = data_samples.shape

    # data_set basic info
    print '+------------------------------------------+'
    print 'Basic Info About Data_Set'
    print 'feature dim is', dim_
    print 'total  rows is', num_samples
    print 'size of object is', format(sys.getsizeof(data_samples)/1024.0/1024.0, '0.2f'), 'MB'
    
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
    iteration_ = 1000

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

