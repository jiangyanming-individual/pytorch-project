# author:Lenovo
# datetime:2023/2/7 15:53
# software: PyCharm
# project:pytorch项目

import torch

x=torch.arange(4, dtype=torch.float32,requires_grad=True)
y= 2 * torch.dot(x,x)
print(y)

y.backward() #反向求导；
print(x.grad)

print("===============")
#累计梯度
x.grad.zero_()  #梯度清零；
y=x.sum()
print(y)
y.backward()
print(x.grad)

print("================")
# print(x.grad.zero_)
x.grad.zero_()
y=x * x
print("y.sum():",y.sum())
y.sum().backward()
print(x.grad)


print("*************************")
x.grad.zero_()

y=x * x
u=y.detach()
z= u * x

z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)











