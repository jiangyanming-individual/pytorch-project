# author:Lenovo
# datetime:2023/2/6 23:43
# software: PyCharm
# project:pytorch项目


import torch

# x = torch.tensor(3.0)
# y = torch.tensor(2.0)
# print(x + y, x * y, x / y, x**y)


A=torch.arange(20).reshape(5,4)
print(A)
print(A.T) #

x=torch.arange(4,dtype=torch.float32)
y=torch.ones(4,dtype=torch.float32)
print(x)
print(y)
print(torch.dot(x,y)) #点乘；先乘再相加；
print(torch.sum(x*y))






