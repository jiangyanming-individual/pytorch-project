# author:Lenovo
# datetime:2023/4/6 17:14
# software: PyCharm
# project:pytorch项目


import torch
x1 = torch.tensor([[1, 2, 3],
                    [4, 5, 6]])

x2 = torch.tensor([[11, 12, 13],
                    [14, 15, 16]])

x3 = torch.tensor([[21, 22],
                    [23, 24]])

out1=torch.cat([x1,x2,x3],dim=-1) #(-1：横着拼接)
out2=torch.cat([x1,x2,x3],dim=1) #(1：横着拼接)
out3=torch.cat([x1,x2],dim=0) #(0：竖着拼接)


print(out1.shape)
print("out1:\n",out1)
print("out2:\n",out2)
print("out3:\n",out3)
