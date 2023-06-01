# author:Lenovo
# datetime:2023/2/8 20:14
# software: PyCharm
# project:pytorch项目


import torch
x=torch.arange(4.0,requires_grad=True)
print(x)

# x的第一个函数：
y= 2 * torch.dot(x,x) #点击，先相乘再相加；是一个标量(一个数)
print("y1:",y) #y1: tensor(28., grad_fn=<MulBackward0>)
y.backward() #反向传播函数自动求导；
print(x.grad) #求x tensor([ 0.,  4.,  8., 12.])

x.grad== 4 * x

#第二个x 的函数
x.grad.zero_(); #梯度清零；
y=x.sum()   # y是一个标量 ，
print("y2:",y)   #y2: tensor(6., grad_fn=<SumBackward0>)
y.backward()
print(x.grad) #tensor([1., 1., 1., 1.])

# x的第三个函数：
x.grad.zero_()
y=x * x  #y是一个向量
print("y3:",y)#y3: tensor([0., 1., 4., 9.], grad_fn=<MulBackward0>)
y.sum().backward() #先求和再求导数;
print(x.grad) #tensor([0., 2., 4., 6.])



















