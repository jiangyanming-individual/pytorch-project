# author:Lenovo
# datetime:2023/2/6 23:35
# software: PyCharm
# project:pytorch项目


import torch

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
#连接张量：
print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))

#计算相等：
print(X==Y)

#求和
print(X.sum())
#转为
A=X.numpy()
B=torch.tensor(A)
print(type(A))
print(type(B))

print()