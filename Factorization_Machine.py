#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : Ryan Fan 
@E-Mail  : ryanfan0528@gmail.com
@Version : v1.0
"""

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt,pow
import itertools
import math
from random import random,shuffle,uniform,seed
import pickle
import sys

"""
lr      : learning rate
momentum: momentum of sgd
nesterov: using nesterov momentum or not
adam    : using adam as optimizer
dropout : dropout rate
l2      : l2 norm for linear weights
l2_fm   : l2 norm for latent weights
l2_bias : l2 norm for bias 
task    : 'r' for regression, 'c' for classification
n_components: dimension of latent
nb_epoch    : rounds to train
interaction : is set to False, it becomes a normal glm(Generalized Linear Model)
no_norm     : normalize inputs or not,default True
"""

"""
libsvm 数据格式如下：(第一列是 labels，后面是依次的 features，空缺表示 null)

+1 4:-0.320755 
-1 1:0.583333 2:-1 3:0.333333
+1 1:0.166667 2:1 3:-0.333333 4:-0.433962  
-1 1:0.458333 3:1 4:-0.358491 

***使用libsvm的好处是对于稀疏的数据能够节省空间。***
"""

def data_generator(path,no_norm=False,task='c'):
    data = open(path,'r')
    for row in data:
        row = row.strip().split(" ")
        y = float(row[0])
        row = row[1:]
        x = []
        for feature in row:
            feature = feature.split(":")
            idx = int(feature[0])
            value = float(feature[1])
            x.append([idx,value])

        if not no_norm:
            r = 0.0
            for i in range(len(x)):
                r+=x[i][1]*x[i][1]
            for i in range(len(x)):
                x[i][1] /=r

        # 对分类问题，用-1和1作为label
        if task=='c':
            if y==0.0:
                y = -1.0

        # 返回的是一个generator object
        yield x,y

# 交叉项
def dot(u,v):
    u_v = 0.
    len_u = len(u)
    for idx in range(len_u):
        uu = u[idx]
        vv = v[idx]
        u_v+=uu*vv
    return u_v

# calculate mse loss 
def mse_loss_function(y,p):
    return (y - p)**2

# calculate mae loss
def mae_loss_function(y,p):
    y = exp(y)
    p = exp(p)
    return abs(y - p)

# calculate log loss
def log_loss_function(y,p):
    return -(y*log(p)+(1-y)*log(1-p))

# calculate exp loss
def exponential_loss_function(y,p):
    return log(1+exp(-y*p))

# sigmoid function
def sigmoid(inX):
    return 1/(1+exp(-inX))

# 对sigmoid设了一个bounded
def bounded_sigmoid(inX):
    return 1. / (1. + exp(-max(min(inX, 35.), -35.)))

# FM with SGD: momentum or nesterov or adam
class SGD(object):
    def __init__(self,lr=0.001,momentum=0.9,nesterov=True,adam=False,l2=0.0,l2_fm=0.0,l2_bias=0.0,ini_stdev= 0.01,dropout=0.5,task='c',n_components=4,nb_epoch=5,interaction=False,no_norm=False):
        self.W = []
        self.V = []        
        self.bias = uniform(-ini_stdev, ini_stdev)
        self.n_components=n_components
        self.lr = lr
        self.l2 = l2
        self.l2_fm = l2_fm
        self.l2_bias = l2_bias
        self.momentum = momentum
        self.nesterov = nesterov
        self.adam = adam
        self.nb_epoch = nb_epoch
        self.ini_stdev = ini_stdev
        self.task = task
        self.interaction = interaction
        self.dropout = dropout
        self.no_norm = no_norm
        if self.task!='c':
            # self.loss_function = mse_loss_function
            self.loss_function = mae_loss_function
        else:
            self.loss_function = exponential_loss_function
            # self.loss_function = log_loss_function

    def preload(self,train,test):
        train = data_generator(train,self.no_norm,self.task) # 返回的是一个generator object
        dim = 0
        count = 0
        for x,y in train:
            for i in x:
                idx,value = i
                if idx >dim:
                    dim = idx
            count+=1
        print('Training samples:',count)
        test = data_generator(test,self.no_norm,self.task)
        count=0
        for x,y in test:
            for i in x:
                idx,value = i
                if idx >dim:
                    dim = idx
            count+=1
        print('Testing samples:',count)
        
        dim = dim+1
        print("Number of features:",dim)

        self.W = [uniform(-self.ini_stdev, self.ini_stdev) for _ in range(dim)]
        self.Velocity_W = [0.0 for _ in range(dim)]
        

        self.V = [[uniform(-self.ini_stdev, self.ini_stdev) for _ in range(self.n_components)] for _ in range(dim)]
        self.Velocity_V = [[0.0 for _ in range(self.n_components)] for _ in range(dim)]

        self.Velocity_bias = 0.0

        self.dim = dim

    def adam_init(self):
        self.iterations = 0
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon=1e-8
        self.decay = 0.
        self.inital_decay = self.decay 

        dim =self.dim

        """
        此处使用下划线的含义解释：
        - 下划线作为临时性的名称使用。
        - 并且不会在后面再次用到该名称。
        - 其实很傻，可以直接用：self.m_W = [0.0] * dim
        """
        self.m_W = [0.0 for _ in range(dim)]
        self.v_W = [0.0 for _ in range(dim)]

        self.m_V = [[0.0 for _ in range(self.n_components)] for _ in range(dim)]
        self.v_V = [[0.0 for _ in range(self.n_components)] for _ in range(dim)]

        self.m_bias = 0.0
        self.v_bias = 0.0

    # 更新bias和非交互项。如果有必要则调用更新一阶交互项的function
    def adam_update(self,lr,x,residual):
        if 0.<self.dropout<1.:
            self.droupout_x(x)
        
        lr = self.lr
        if self.inital_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1

        lr_t = lr * sqrt(1. - pow(self.beta_2, t)) / (1. - pow(self.beta_1, t))
        
        for sample in x:
            idx,value = sample
            g = residual*value

            m = self.m_W[idx]
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

            v = self.v_W[idx]
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * (g**2)

            p = self.W[idx]
            p_t = p - lr_t *m_t / (sqrt(v_t) + self.epsilon) 

            if self.l2>0:
                pt = pt - lr_t*self.l2*p

            self.m_W[idx] = m_t
            self.v_W[idx] = v_t
            self.W[idx] = p_t

        # 如果是FM，就更新交互项的参数
        if self.interaction:
            self._adam_update_fm(lr_t,x,residual)

        # 使用adam更新偏置项的参数
        m = self.m_bias
        m_t = (self.beta_1 * m) + (1. - self.beta_1)*residual

        v = self.v_bias
        v_t = (self.beta_2 * v) + (1. - self.beta_2)*(residual**2)

        p = self.bias
        p_t = p - lr_t * m_t / (sqrt(v_t) + self.epsilon)
        if self.l2_bias>0:
            pt = pt - lr_t * self.l2_bias*p

        self.m_bias = m_t
        self.v_bias = v_t
        self.bias = p_t

        self.iterations+=1

    # 使用adam更新一阶交互项
    def _adam_update_fm(self,lr_t,x,residual):
        len_x = len(x)
        sum_f_dict = self.sum_f_dict
        n_components = self.n_components
        for f in range(n_components):
            for i in range(len_x):
                idx_i,value_i = x[i]
                v = self.V[idx_i][f]
                sum_f = sum_f_dict[f]
                g = (sum_f*value_i - v *value_i*value_i)*residual

                m = self.m_V[idx_i][f]
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

                v = self.v_V[idx_i][f]
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * (g**2)

                p = self.V[idx_i][f]
                p_t = p - lr_t * m_t / (sqrt(v_t) + self.epsilon)

                if self.l2_fm>0:
                    pt = pt - lr_t * self.l2_fm*p

                self.m_V[idx_i][f] = m_t
                self.v_V[idx_i][f] = v_t
                self.V[idx_i][f] = p_t

    # 随机的删除sample在某个feature上的取值
    def droupout_x(self,x):
        new_x = []
        for i, var in enumerate(x):
            if random() > self.dropout:
                del x[i]

    def _predict_fm(self,x):
        len_x = len(x)
        n_components = self.n_components
        pred = 0.0
        self.sum_f_dict = {}
        for f in range(n_components):
            sum_f = 0.0
            sum_sqr_f = 0.0
            for i in range(len_x):
                idx_i,value_i = x[i]
                d = self.V[idx_i][f] * value_i
                sum_f +=d
                sum_sqr_f +=d*d
            pred+= 0.5 * (sum_f*sum_f - sum_sqr_f);
            self.sum_f_dict[f] = sum_f
        return pred

    def _predict_one(self,x):
        pred = self.bias
        # pred = 0.0
        for idx,value in x:
            pred+=self.W[idx]*value
        
        if self.interaction:
            pred+=self._predict_fm(x)

        # if self.task=='c':
        #     pred = bounded_sigmoid(pred)
        return pred

    # 使用momentum或nesterov更新bias和非交互项
    def update(self,lr,x,residual):
        if 0.<self.dropout<1.:
            self.droupout_x(x)

        for sample in x:
            idx,value = sample
            grad = residual*value
            self.Velocity_W[idx] = self.momentum * self.Velocity_W[idx] - lr * grad
            if self.nesterov:
                # nesterov 比 momentum 多使用一次Velocity
                self.Velocity_W[idx] = self.momentum * self.Velocity_W[idx] - lr * grad
            self.W[idx] = self.W[idx] + self.Velocity_W[idx] - self.l2*self.W[idx]
            
        if self.interaction:
            self._update_fm(lr,x,residual)

        self.Velocity_bias = self.momentum*self.Velocity_bias - lr*residual
        if self.nesterov:
            self.Velocity_bias = self.momentum*self.Velocity_bias - lr*residual
        self.bias = self.bias +self.Velocity_bias - self.l2_bias*self.bias

    # 使用momentum或nesterov更新一阶交互项
    def _update_fm(self,lr,x,residual):
        len_x = len(x)
        sum_f_dict = self.sum_f_dict
        n_components = self.n_components
        for f in range(n_components):
            for i in range(len_x):
                idx_i,value_i = x[i]
                sum_f = sum_f_dict[f]
                v = self.V[idx_i][f]
                grad = (sum_f*value_i - v *value_i*value_i)*residual
                
                self.Velocity_V[idx_i][f] = self.momentum * self.Velocity_V[idx_i][f] - lr * grad
                if self.nesterov:
                    self.Velocity_V[idx_i][f] = self.momentum * self.Velocity_V[idx_i][f] - lr * grad
                self.V[idx_i][f] = self.V[idx_i][f] + self.Velocity_V[idx_i][f] - self.l2_fm*self.V[idx_i][f]

    def predict(self,path,out):
        data = data_generator(path,self.no_norm,self.task)
        y_preds =[]
        with open(out, 'w') as outfile:
            ID = 0
            outfile.write('%s,%s\n' % ('ID', 'target'))
            for d in data:
                x,y = d
                p = self._predict_one(x)
                outfile.write('%s,%s\n' % (ID, str(p)))
                ID+=1

    def validate(self,path):
        data = data_generator(path,self.no_norm,self.task)
        loss = 0.0
        count = 0.0

        for d in data:
            x,y = d
            p = self._predict_one(x)
            loss+=self.loss_function(y,p)
            count+=1
        return loss/count

    def save_weights(self):
        weights = []
        weights.append(self.W)
        weights.append(self.V)
        weights.append(self.bias)
        weights.append(self.Velocity_W)
        weights.append(self.Velocity_V)
        weights.append(self.dim)

        """
        为什么要pickle数据：
        - 被pickle的数据，在被多次reload时，不需要重新去计算得到这些数据，这样节省计算机资源，如果你不pickle，你每调用一次数据，就要计算一次；
        - 类似游戏的Save/Load；
        """
        pickle.dump(weights,open('sgd_fm.pkl','wb'))

    def load_weights(self):
        weights = pickle.load(open('sgd_fm.pkl','rb'))
        self.W = weights[0]
        self.V = weights[1]
        self.bias = weights[2]
        self.Velocity_W = weights[3]
        self.Velocity_V = weights[4]
        self.dim = weights[5]
        
    def train(self,path,valid_path = None,in_memory=False):
        start = datetime.now()
        lr = self.lr
        if self.adam:
            self.adam_init()
            self.update = self.adam_update

        if in_memory:
            data = data_generator(path,self.no_norm,self.task)
            """
            使用列表解析的方式比使用列表递推（for loop + xx.append）快一倍
            """
            data = [d for d in data] 
            
        best_loss = 999999
        best_epoch = 0
        for epoch in range(1,self.nb_epoch+1):
            if not in_memory:
                data = data_generator(path,self.no_norm,self.task)
            train_loss = 0.0
            train_count = 0
            for x,y in data:
                p = self._predict_one(x)
                if self.task!='c':                    
                    residual = -(y-p)
                else:
                    residual = -y*(1.0-1.0/(1.0+exp(-y*p)));
                    # residual = -(y-p)

                self.update(lr,x,residual)
                if train_count%50000==0:
                    if train_count ==0:
                        print '\ttrain_count: %s, current loss: %.6f'%(train_count,0.0)
                    else:
                        print '\ttrain_count: %s, current loss: %.6f'%(train_count,train_loss/train_count)

                train_loss += self.loss_function(y,p)
                train_count += 1

            epoch_end = datetime.now()
            duration = epoch_end-start
            
            if valid_path:
                valid_loss = self.validate(valid_path)
                print('Epoch: %s, train loss: %.6f, valid loss: %.6f, time: %s'%(epoch,train_loss/train_count,valid_loss,duration))
                if valid_loss<best_loss:
                    best_loss = valid_loss
                    self.save_weights()
                    print 'save_weights'
            else:
                print('Epoch: %s, train loss: %.6f, time: %s'%(epoch,train_loss/train_count,duration))

def main():
    seed(313)
    path = "H:\\Allstate\\"

    sgd = SGD(lr=0.002,momentum=0.9,adam=True,nesterov=True,dropout=0.3,l2=0.0,l2_fm=0.0,task='r',n_components=4,nb_epoch=63,interaction=True,no_norm=False)
    sgd.preload(path+'X_cat_oh_high_order.svm',path+'X_t_cat_oh_high_order.svm')
    sgd.train(path+'X_train_cat_oh_high_order.svm',path+'X_test_cat_oh_high_order.svm',in_memory=False)
    sgd.predict(path+'X_test_cat_oh_high_order.svm',out='valid.csv')
    sgd.predict(path+'X_t_cat_oh_high_order.svm',out='out.csv')

if __name__ == '__main__':
    main()