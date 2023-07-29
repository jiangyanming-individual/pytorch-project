# author:Lenovo
# datetime:2023/7/9 8:47
# software: PyCharm
# project:pytorch项目



from imblearn.over_sampling import SMOTE
from collections import Counter

# 原始数据集
# X = [[1, 2], [3, 4], [1, 2], [3, 4]]
# y = [0, 0, 1, 1]
#
# print("原始数据集：")
# print("X:", X)
# print("y:", y)
#
# # 使用SMOTE算法进行数据生成
# smote = SMOTE()
# X_resampled, y_resampled = smote.fit_resample(X, y)
#
# print("生成后的数据集：")
# print("X_resampled:", X_resampled)
# print("y_resampled:", y_resampled)
#
# # 查看生成后的数据集分布
# print("生成后的数据集分布：")
# print(Counter(y_resampled))



from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# summarize class distribution


print("X:",X)
print("X shape",X.shape)
print("y:",y)
counter = Counter(y)
# print(counter)

# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()


#学习分类 ,少数类型过采样的操作

oversample=SMOTE()
X,y=oversample.fit_resample(X,y)
counter=Counter(y)
print(counter)

for label,_ in counter.items():

    row_ix=where(y==label)[0]
    # print(row_ix)
    pyplot.scatter(X[row_ix,0],X[row_ix,1],label=str(label))
pyplot.legend()
pyplot.show()


