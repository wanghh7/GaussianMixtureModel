# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:18:17 2021

@author: wanghaohui
"""
import numpy as np
import sys
sys.path.append("..")
from AL.gmm import *

# 第一簇的数据
num1, mu1, var1 = 400, [0.5, 0.5], [1, 3]
X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
# 第二簇的数据
num2, mu2, var2 = 600, [5.5, 2.5], [2, 2]
X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
# 第三簇的数据
num3, mu3, var3 = 1000, [1, 7], [6, 2]
X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
# 合并在一起
X = np.vstack((X1, X2, X3))

# 设定参数值
k = 3
gmm_model = GMM(k, tol, maxiter)
gmm_model.fit(X)
gamma, alpha = gmm_model.gamma, gmm_model.alpha