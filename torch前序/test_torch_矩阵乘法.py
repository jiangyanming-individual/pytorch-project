# author:Lenovo
# datetime:2023/2/7 15:37
# software: PyCharm
# project:pytorch项目
import torch

A=torch.arange(12,dtype=torch.float32).reshape(3,4)
B=torch.ones((4,3),dtype=torch.float32)

print(torch.mm(A,B))

X=torch.normal(mean=1,std=2,size=(3,4))
print(X)
