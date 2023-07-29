# author:Lenovo
# datetime:2023/7/9 9:50
# software: PyCharm
# project:pytorch项目


"""
KNN紧邻值的选取对于AUC的影响：
"""
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# define model

k_values = [1, 2, 3, 4, 5, 6, 7]

for k in k_values:

    model = DecisionTreeClassifier()
    #使用少数样本欠采样的方式：
    over=SMOTE(sampling_strategy=0.1,k_neighbors=k)
    under=RandomUnderSampler(sampling_strategy=0.5)
    steps=[('over',over),('under',under),('model',model)]
    piplines=Pipeline(steps)

    # 10kfold
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(piplines, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('>K=%d ,Mean ROC AUC:%.3f' %(k,mean(scores)))

