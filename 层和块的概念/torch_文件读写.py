# author:Lenovo
# datetime:2023/2/15 15:51
# software: PyCharm
# project:pytorch项目


import torch
from torch import nn

#存储X的张量；
X=torch.arange(4)
torch.save(X,'X-file')

X2=torch.load('X-file')
print(X2)

#创造一个元组类型：
Y=torch.zeros(4)
torch.save([X,Y],'X-file')
X2,Y2=torch.load('X-file')
print((X2,Y2))


#字典类型的张量
mydict={'X':X,'Y':Y}
torch.save(mydict,'mydict')
mydict2=torch.load('mydict')
print(mydict2)









