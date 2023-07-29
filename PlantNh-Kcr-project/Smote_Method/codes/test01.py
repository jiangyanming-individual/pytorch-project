# author:Lenovo
# datetime:2023/7/9 10:43
# software: PyCharm
# project:pytorch项目


from imblearn.over_sampling import SMOTE
from Bio.Seq import Seq
from collections import Counter

# 原始不平衡的氨基酸序列样本
seqs = ['ARNAL', 'ASPAR', 'RSDKA', 'MMEKK', 'ARGLL','KKKAAC','DDARC','AANCD','ADCRR','RRADD']
y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  # 对应的类别标签，0代表少数类，1代表多数类

# 将氨基酸序列转换为特征向量（例如one-hot编码）
X = []
for seq in seqs:
    seq_obj = Seq(seq)
    X.append([int(seq_obj.count('A')), int(seq_obj.count('R')),
              int(seq_obj.count('N')), int(seq_obj.count('D')),
              int(seq_obj.count('C'))])
    # print("X:",X)
print(type(X))

print("原始氨基酸序列样本：")
print("X:", X)
print("y:", y)

# 使用SMOTE算法生成平衡的氨基酸序列样本
smote = SMOTE(k_neighbors=2,random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("生成后的氨基酸序列样本：")
print("X_resampled:", X_resampled)
print("y_resampled:", y_resampled)

# 查看生成后的样本分布
print("生成后的样本分布：")
print(Counter(y_resampled))