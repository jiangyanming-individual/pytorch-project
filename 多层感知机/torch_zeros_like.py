# author:Lenovo
# datetime:2023/2/12 15:56
# software: PyCharm
# project:pytorch项目


import torch

X=torch.arange(12).reshape(3,4)
print(X)
Y=torch.zeros_like(X)
print(Y)