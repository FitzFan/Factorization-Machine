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

5、FM的接口，建议选用pywFM和pyFM，二者区别不是很大
- pyFM :https://github.com/coreylynch/pyFM/blob/master/pyfm/pylibfm.py
- pywFM:https://github.com/jfloff/pywFM
- 二者的区别：http://mufool.com/2017/11/20/fm/
pywFM涉及到的主要参数：（pyFM也差不多）
	- k2:2阶交互项，每个特征对应的latent factor的长度。

6、FFM和FM相比：
- 最大的区别：假设样本的 n 个特征属于 f 个field，那么FFM的二次项有 nf 个隐向量。而在FM模型中，每一维特征的隐向量只有一个。FM可以看作FFM的特例
- LR、poly、FM、FFM在做特征交叉的区别：参见FFM进行特征交叉的实例.pdf
- 用FFM做特征组合的本质：特征_1与特征_2交叉的结果为：特征_1在特征_2所属field的latent factor 与 特征_2在特征_1所属field的latent factor 进行内积。
- 暂时未找到比较好的实现FFM的python接口。所以不知道具体的实现方式，也不知道参数怎么控制结果。比如field怎么定？
- 面试时，坦言只了解过原理，未实现过。


7、基于fastFM包实现FM算法，分别使用SGD、ALS、MCMC三种优化算法
- 使用demo：https://blog.csdn.net/jiangda_0_0/article/details/77510029

8、ALS填坑
- 资料：https://www.jianshu.com/p/9a584bba1c68、https://blog.csdn.net/m0_37788308/article/details/78196674
- ALS有一个假设是打分矩阵是低秩矩阵。个人认为这应该是用MF做CF的假设才对。
- ALS本质上是一个optimization algorithm，地位和Gradient Descent一样；
- ALS一般用在矩阵分解。可以理解为：矩阵分解是模型（决定loss function的样子），ALS和SGD是optimizer（帮助寻找最优解的位置）
- ALS和SMO以及EM算法的的思想有点类似，即可以借助对SMO的理解来帮助推导ALS的解。
- ALS的优点：收敛速度快、对初始值不敏感；
- ALS的缺点：需要对user和item的latent factor matrix求逆矩阵。
- ALS的进阶变形：ALS-WR
-- ALS-WR解决的问题背景：
	多情况下，用户没有明确反馈对商品的偏好，也就是没有直接打分，我们只能通过用户的某些行为来推断他对商品的偏好。比如，在电视节目推荐的问题中，对电视节目收看的次数或者时长，这时我们可以推测次数越多，看得时间越长，用户的偏好程度越高，但是对于没有收看的节目，可能是由于用户不知道有该节目，或者没有途径获取该节目，我们不能确定的推测用户不喜欢该节目。ALS-WR通过置信度权重来解决这些问题：对于更确信用户偏好的项赋以较大的权重，对于没有反馈的项，赋以较小的权重。
-- 资料：https://www.jianshu.com/p/9a584bba1c68
-- 本质就是对loss function进行修正，使其更符合实际情况。

"""


