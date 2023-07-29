# author:Lenovo
# datetime:2023/7/9 9:07
# software: PyCharm
# project:pytorch项目

"""
用于平衡数据集的SMOTE
少数类可以用于过采样，
多数类可以用于过采样
"""

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where

X,y=make_classification(n_samples=10000,n_features=2,n_redundant=0,
                        n_clusters_per_class=1,weights=[0.99],flip_y=0,random_state=1
                        )
counter = Counter(y)
print(counter)

#少数样本过采样
over=SMOTE(sampling_strategy=0.1)

#对于多数样本欠采样
under=RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]

#管道执行流：
piplines=Pipeline(steps=steps)
X,y=piplines.fit_resample(X,y)


counter = Counter(y)
print(counter)


for label,_ in counter.items():

    row_ix=where(y == label)[0]
    pyplot.scatter(X[row_ix,0],X[row_ix,1],label=str(label))

pyplot.legend()
pyplot.show()
