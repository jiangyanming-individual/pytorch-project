# author:Lenovo
# datetime:2023/2/19 19:11
# software: PyCharm
# project:pytorch项目


import torch

X=torch.tensor([[1,2,3,4],[2,3,4,5],[4,6,7,8]])
print(X.shape)#torch.Size([3, 4])
print(X.size()) #torch.Size([3, 4])

print(X.shape[0]) #3 行数
print(X.shape[1]) #4 列数
print(X.size(0)) #3 行数
print(X.size(1)) #4  列数


