# author:Lenovo
# datetime:2023/4/19 9:34
# software: PyCharm
# project:pytorch项目


import math
import torch
from torch import nn
from d2l import torch as d2l

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
# print(data_iter)
embed=nn.Embedding(num_embeddings=20,embedding_dim=4)

# print(f'Parameter embedding_weight({embed.weight.shape},dtype={embed.weight.dtype})')
X=torch.tensor([[1,2,3],[4,5,6]])
res=embed(X) #torch.Size([2, 3, 4]) 经过embedding 变为 2 *(3 *4)

# print(res.shape)
# print(res)

def skip_gram(center,contexts_and_negatives,embed_v,embed_u):
    #这两个变量首先通过嵌入层从词元索引转换成向量，然后它们的批量矩阵相乘
    v=embed_v(center) #(2,1,4==>embedding_dim)
    u=embed_u(contexts_and_negatives) #(2,4,4==>embedding_dim)

    #转化一下矩阵： u.permute(0,2,1) ===>2 *4(embedding_dim) *4
    pred=torch.bmm(v,u.permute(0,2,1)) #输出中的每个元素是中心词向量和上下文或噪声词向量的点积。
    return pred

skip_gram_res=skip_gram(torch.ones((2,1),dtype=torch.long),torch.ones((2,4),dtype=torch.long),embed,embed)
print(skip_gram_res.shape)


#训练：
class SigmoidBCELoss(nn.Module):

    def __init__(self):
        super(SigmoidBCELoss, self).__init__()


    def forward(self,inputs,target,mask=None):
        out=nn.functional.binary_cross_entropy_with_logits(inputs,target,weight=mask,reduction="none")

        print(out)
        return out.mean(dim=1)

loss = SigmoidBCELoss()

pred=torch.tensor([[1.1, -2.2, 3.3, -4.4],[1.1, -2.2, 3.3, -4.4]])
label=torch.tensor([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]])
# print(pred)
mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
print(mask.shape[1]) #===>4

# print("mask.sum(axis=1):",mask.sum(axis=1)) #torch.Size([4,2])
# loss(pred,label,mask) * mask.shpae[1] /mask.sum(axis=1)

#
# def sigmd(x):
#     return -math.log(1 / (1 + math.exp(-x)))


embed_size=100
net=nn.Sequential(
    nn.Embedding(num_embeddings=len(vocab),embedding_dim=embed_size),
    nn.Embedding(num_embeddings=len(vocab),embedding_dim=embed_size)
)

def train(net,data_iter,lr,num_epochs,device):

    #初始化参数：
    def init_weight(m):
        if type(m) ==nn.Embedding:
            nn.init.xavier_uniform_(m.weight) #均匀分布

    net.apply(init_weight)
    net=net.to(device) #转移到device运算数据：
    #优化器：
    opt=torch.optim.Adam(net.parameters(),lr=lr)
    #画图
    animator=d2l.Animator(xlabel='epoch',ylabel='loss',xlim=[1,num_epochs])

    metric=d2l.Accumulator(2)
    for epoch in range(num_epochs):

        timer,num_bacthes=d2l.Timer(),len(data_iter)
        for i,batch in enumerate(data_iter):
            opt.zero_grad()
            #取数据：
            center,context_negatives,mask,label=[data.to(device) for data in batch]

            pred=skip_gram(center,context_negatives,net[0],net[1])
            l=(loss(pred.reshape(label.shape).float(),label.float(),mask)
                * mask.shape[1] / mask.sum(axis=1)
               )

            l.sum().backward()
            opt.step()
            metric.add(l.sum(),l.numel())
            if (i+1) %(num_bacthes // 5) ==0 or i == num_bacthes -1:
                animator.add(epoch +(i+1) /num_bacthes,(metric[0]/metric[1],))

    print(f'loss {metric[0] /metric[1]}:.3f,{metric[1] / timer.stop():.1f} ')

lr,num_epochs=0.002,5
train(net,data_iter=data_iter,lr=lr,num_epochs=num_epochs,device='cpu')