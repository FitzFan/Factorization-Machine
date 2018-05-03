#!/usr/bin/env python
#coding:utf-8



"""
1、FM初探
- 非常棒的学习资料：https://plushunter.github.io/2017/07/13/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E7%B3%BB%E5%88%97%EF%BC%8826%EF%BC%89%EF%BC%9A%E5%9B%A0%E5%AD%90%E5%88%86%E8%A7%A3%E6%9C%BA%EF%BC%88FM%EF%BC%89%E4%B8%8E%E5%9C%BA%E6%84%9F%E7%9F%A5%E5%88%86%E8%A7%A3%E6%9C%BA%EF%BC%88FFM%EF%BC%89/
- FM的优势：
-- 矩阵分解的算法，都有一个毛病：特征扩展性较差。
-- FM等算法，相当于是实现了轻松扩展的SVD++；

2、FM和LR的区别：
- 参见：https://www.zhihu.com/question/27043630
- LR是从组合特征的角度去描述单特征之间的交互组合，Factorization Machine实际上是从模型的角度来做的。即FM中特征的交互是模型参数的一部分。
- FM能很大程度上避免了数据系数行造成参数估计不准确的影响，为什么?
	a) FM是通过MF的思想，基于latent factor，来降低交叉项参数学习不充分的影响；
	b) 具体而言，两个交互项的参数学习，是基于K维的latent factor。
	c) 每个特征的latent factor是通过它与其它(n-1)个特征的latent factor交互后进行学习的，这就大大的降低了因稀疏带来的学习不足的问题。
	d) latent factor学习充分了，交互项（两个latent factor之间的内积）也就学习充分了。
	e) 即FM学习的交互项的参数是单特征的隐向量。
3、一般情况下，FM的以二阶多项式模型（degree=2时）居多，但也可以将M模型的degree设为3，此时就是考虑User-Ad-Context三个维度特征之间的关系。

4、有种说法是：FM模型称为多项式的广义线性模型，个人认为是不准确的：
- FM和线性模型有着很大的不同：线性模型是直接通过数据学习特征的参数，而FM是基于MF学习特征的latent factor。

5、FM的接口，建议选用pywFM
主要参数：
	- k2:2阶交互项，每个特征对应的latent factor的长度。
	- **1**对比一下SGD、ALS、MCMC的区别和优劣势
	- **2**学习ALS优化算法


6、FFM和FM相比：
- 最大的区别：假设样本的 n 个特征属于 f 个field，那么FFM的二次项有 nf个隐向量。而在FM模型中，每一维特征的隐向量只有一个。FM可以看作FFM的特例
- LR、poly、FM、FFM在做特征交叉的区别：参见FFM进行特征交叉的实例.pdf
- 用FFM做特征组合的本质：特征_1与特征_2交叉的结果为：特征_1在特征_2所属field的latent factor 与 特征_2在特征_1所属field的latent factor 进行内积。
- 暂时未找到比较好的实现FFM的python接口。所以不知道具体的实现方式，也不知道参数怎么控制结果。比如field怎么定？
- 面试时，坦言只了解过原理，未实现过。
"""


