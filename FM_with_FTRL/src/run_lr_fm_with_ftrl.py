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
import traceback
import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from FM_FTRL import FM
from LR_FTRL import LR, evaluate_model, get_auc
from preprocessing import data_preprocess
from send_email_src import send_email


"""
- algo desc:
    - DIY python implementation of Factorization Machines and Logistic Regression
    - select model to train
    - optimizer is FTRL
    - huge_data_set one-hot algorithm
    - Mail Notifier
"""

"""
- 优化的起点在于监控自己的算法：top -u peng.fan
- 已做优化：
    - 使用gabage collect 回收内存；
    - read_csv()时，指定dtype。 【done，效率提升80%左右，内存节省200%以上】 ^ ^
    - 使用pd.to_numeric()来优化特征类型，尝试后发现会起到反作用（内存占用提高了300%） - -
    
- 待做的优化：
    - 相关参数写在conf.py
        - 发邮件信息；
        - 数据路径；
    - maybe set chunksize parameter can be better? link: https://yashuseth.blog/2017/12/14/how-to-one-hot-encode-categorical-variables-of-a-large-dataset-in-python/
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

def preprocess(raw_data_path, need_label_encode_dic, need_one_hot_dic):
    # 删除中间数据文件夹的数据
    del_cmd = 'rm ../temp_data/*'
    del_out = commands.getstatusoutput(del_cmd)

    if str(del_out[0]) == '0':
        print 'Succeed to flush all ../temp_data/*'
        print '+------------------------------------------+'
    else:
        print str(del_out[1])

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
        data_samples = pd.read_csv(raw_data_path, sep=',', 
                                   usecols=[need_label_encode_dic[col_name]], 
                                   error_bad_lines=False, dtype='object', engine='c')

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
        data_samples = pd.read_csv(raw_data_path, sep=',', 
                                   usecols=[need_one_hot_dic[col_name]], 
                                   error_bad_lines=False, dtype='object', engine='c')

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
    
def train_model(X_train, X_test, y_train, y_test, hyper_params, iteration_, lr_tag=True, fm_tag=True): 
    # time clock
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 重启clock
    start_train = time.clock()

    # 新建数据盒子
    what_to_do_list = []
    have_done_list = []

    print '+------------------------------------------+'
    print 'Begin To Train Model:', start_time
    
    # select model to train
    if lr_tag:
        # train models
        lr = FTRL(base="lr", args_parse=hyper_params, iteration=iteration_)
        lr.fit(X_train, y_train)

        # test the unseen samples
        test_preds_lr = lr.predict(X_test)
        test_error_lr = evaluate_model(test_preds_lr, y_test)
        test_auc_lr = roc_auc_score(y_true=y_test, y_score=test_preds_lr)
        my_test_auc_lr = get_auc(scores=test_preds_lr, labels=y_test)

        # print train model info
        print("logistic regression test error: %.2f%%" % test_error_lr)
        print("logistic regression test auc: ", test_auc_lr)
        print("logistic regression-my test auc: ", my_test_auc_lr)
        what_to_do_list.append("logistic regression-test error")
        what_to_do_list.append("logistic regression-test auc")
        what_to_do_list.append("logistic regression-my test auc")
        have_done_list.append(str(test_error_lr))
        have_done_list.append(str(test_auc_lr))
        have_done_list.append(str(my_test_auc_lr))
        print '+------------------------------------------+'

    if fm_tag:
        # train models
        fm = FTRL(base="fm", args_parse=hyper_params, iteration=iteration_)
        params = fm.fit(X_train, y_train)

        # test the unseen samples
        test_preds_fm = fm.predict(X_test)
        test_error_fm = evaluate_model(test_preds_fm, y_test)
        test_auc_fm = roc_auc_score(y_true=y_test, y_score=test_preds_fm)
        my_test_auc_fm = get_auc(scores=test_preds_fm, labels=y_test)

        # print train model info
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
        print '\n'
        print("FM weights: ", params['weights'])
        print("FM bias: ", params['bias'])
        print("FM latent vectors: ", params['V']) 
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
    after_pre_data_path = '../data/all_featrue_after_preprocessing.dat'  # preprocess()处理后生成的文件
    feature_analyse_path = '../out_put/feature_ana.dat' # shell脚本做的特征预分析

    # set features needed to preprocessing
    """
    - dict:
        - key：col_name
        - value:col_num
    - one_hot 和 label_encode只需要做其中一个
    - 坑：shell索引的起点是1，python是0。
    """
    # 读取feature_analyse_data
    feature_ana_frame = pd.read_csv(feature_analyse_path)

    # 过滤出连续型的feature
    cont_feature = feature_ana_frame[feature_ana_frame['col_type'].apply(lambda x:str(x).find('int')!=-1 or str(x).find('float')!=-1)]

    # 连续型特征字典
    continuous_feature_dic = dict(zip(cont_feature['col_name'], cont_feature['python_num']))
    continuous_feature_type = dict(zip(cont_feature['col_name'], cont_feature['col_type']))

    # 手动设置需要预处理的特征字典
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

    # maybe部分连续型变量做了label_encode 或 one_hot
    for key in continuous_feature_dic.keys():
        if key in set(need_label_encode_dic.keys()) | set(need_one_hot_dic.keys()):
            continuous_feature_dic.pop(key)
            continuous_feature_type.pop(key)

    # data preprocessing
    preprocess(raw_data_path, need_label_encode_dic, need_one_hot_dic)

    # 读取少量行的数据
    data_samples = pd.read_csv(after_pre_data_path, sep=',', error_bad_lines=False, engine='c', nrows=24) # 读取所有字段的前24行

    # 保存每一列的数据类型
    col_dtype_dict = {}
    for col in data_samples.columns:
        if col.find('#$#') != -1:
            col_dtype_dict[col] = 'uint8'
        else:
            col_dtype_dict[col] = str(data_samples[col].dtypes)

    # read all preprocessed data
    print '+------------------------------------------+'
    print 'Begin To read all train data set and concat'

    # 指定每个字段的dtype的方式读取csv文件，以节约内存
    cate_samples = pd.read_csv(after_pre_data_path, sep=',', dtype=col_dtype_dict, error_bad_lines=False, engine='c')
    cont_samples = pd.read_csv(raw_data_path, sep=',', 
                               usecols=[continuous_feature_dic[key] for key in continuous_feature_dic], 
                               error_bad_lines=False, engine='c', dtype=continuous_feature_type)
    # 合并data_frame
    data_samples = pd.concat([cate_samples, cont_samples], axis=1)
    categorical_dim = len(cate_samples.columns)
    continuous_dim = len(cont_samples.columns)

    if len(data_samples.columns) != categorical_dim+continuous_dim:
        print '+------------------------------------------+'
        print 'Fail To data concat, please debug.'
        sys.exit()
    else:
        nan_info = data_samples.shape[0] - data_samples.count()
        nan_list = nan_info[nan_info.apply(lambda x: x!=0)]
        if len(nan_list) != 0:
            print '+------------------------------------------+'
            print 'NaN info Of Data_Set'
            print 'Following Features Have NaN value:'
            print nan_list
        else:
            print '+------------------------------------------+'
            print 'No Feature has NaN value'
            print '+------------------------------------------+'

    # gabage collect
    del cate_samples, cont_samples
    gc.collect()

    print 'Begin To read label_ryan.dat'
    target_samples = pd.read_csv(label_data_path, sep=',', error_bad_lines=False, engine='c')

    # 获取数据集基本情况
    num_samples, dim_ = data_samples.shape

    # 计算数据占用内存情况
    mem_use = sum(data_samples.memory_usage(deep=True))
    mem_use = float(mem_use) / 1024 ** 3

    # data_set basic info
    print 'categorical_dim is', categorical_dim
    print 'continuous_dim is ', continuous_dim
    print 'total rows is', num_samples

    # 邮件提醒已读完所有数据
    receivers = ['ryanfan0313@163.com']
    Subject = 'Succeed To Read All Train Data'
    text = '***Data Info As Following: ***\n'
    text += '- feature_data_set Memorry Used: %f GB \n'%(round(mem_use, 2))
    text += '- categorical_dim: %d\n'%(categorical_dim)
    text += '- continuous_dim : %d\n'%(continuous_dim)
    text += '- total data rows: %d' %(num_samples)
    print text
    table_name = ' '
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    send_email_func = send_email(receivers, text, Subject, table_name, date)
    send_email_func.email_plain_text()

    # split all the samples into training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(data_samples, target_samples, test_size=0.2, random_state=313)

    # 4千万的数据有些扛不住，做一个sampling
    X_train = X_train[0:400000]
    X_test  = X_test[0:400000]
    y_train = y_train[0:400000]
    y_test  = y_test[0:400000]

    print '---------------------'
    print 'data samples is', len(X_train)
    print '---------------------'

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

    # select model
    sel_lr = False
    sel_fm = True

    # train model
    train_model(X_train, X_test, y_train, y_test, hyper_params, iteration_, sel_lr, sel_fm)

    """
    还差：save model and test use model
    """

if __name__ == "__main__":
    # 如果出错，将错误信息发送至监控邮箱
    try:
        main()
    except:
        print '+******************************************+'
        print str(traceback.format_exc())
        print '+******************************************+'

        # 设置邮件发送基本信息
        receivers = ['ryanfan0313@163.com']
        Subject = 'Please debug for run_lr_fm_with_ftrl.py'
        text = 'Error Msg As Following: ' + '\n' + str(traceback.format_exc())  # 设置错误信息的格式
        date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        table_name = ''
    
        # 错误代码作为邮件内容，发送邮件
        send_email_func = send_email(receivers, text, Subject, table_name, date)
        send_email_func.email_plain_text()

