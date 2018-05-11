# yiji
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 19:44:19 2018

@author: asus
"""

print(__doc__)
# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
'''To choose parameters of the model, this paper adopted the method of cross validation based on grid search, 
avoiding the arbitrary and capricious behav .
采用基于网格搜索的交叉验证法来选择模型参数，避免了参数选择的盲目性和随意性'''
logistic = linear_model.LogisticRegression() #逻辑回归
pca = decomposition.PCA() #主成分分析
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
# Plot the PCA spectrum
pca.fit(X_digits)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2) #在统计学把方差分解为解释方差（可以明确方差来源的）和未解释方差
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
# Prediction
n_components = [20, 40, 64]
Cs = np.logspace(-4, 4, 3)
# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs)) #dict 字典的添加、删除、修改
estimator.fit(X_digits, y_digits)
plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()
