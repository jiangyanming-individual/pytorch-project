# author:Lenovo
# datetime:2023/2/6 21:04
# software: PyCharm
# project:pytorch项目

import torch
x=torch.arange(12)
print(x)
# print(x.shape)
# print(x.numel()) #张量的大小；
# print(x.reshape([3,4]))
print(torch.zeros([2,3,4]))
print("==================")
print(torch.ones([3,4]))
print("====================")
print(torch.randn([3,4])) #返回一个 3* 4的正态分布；

print(torch.exp(x))


