# 一个基于PyTorch实现的一维卷积胶囊网络示例：
# 这个模型包括三个部分：卷积层、胶囊层和动态路由层。
# 在卷积层中，我们使用一个一维卷积层对输入进行处理。
# 在胶囊层中，我们使用多个卷积胶囊分别对卷积层的输出进行处理，并将它们的输出拼接在一起。
# 最后，在动态路由层中，我们使用动态路由算法来计算每个输出的胶囊向量。
# 在实现中，我们使用了PyTorch提供的softmax和norm函数，
# 以及自定义的_squash函数。我们还使用了nn.ModuleList来管理多个胶囊。
# 为了方便调用，我们提供了一些参数，如卷积核大小、步幅、填充、胶囊数量、胶囊维度和路由算法的迭代次数。

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvCapsNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_capsules, capsule_dim, kernel_size, stride=1, padding=0):
        super(ConvCapsNet1D, self).__init__()

        # 卷积层
        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        # 胶囊层
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.primary_capsules = nn.ModuleList([
            nn.Conv1d(out_channels, capsule_dim, kernel_size=1, stride=1, padding=0) for _ in range(num_capsules)
        ])

        # 动态路由层
        self.iterations = 3
        self.W = nn.Parameter(torch.randn(num_capsules, in_channels, capsule_dim, 1, 1))

    def forward(self, x):
        # 卷积层
        x = self.conv_layer(x)
        batch_size, _, seq_len = x.size()

        # 胶囊层
        u = [capsule(x).view(batch_size, self.capsule_dim, 1, seq_len) for capsule in self.primary_capsules]
        u = torch.cat(u, dim=2)
        u_hat = u.detach()

        # 动态路由层迭代
        for _ in range(self.iterations):
            c = F.softmax(self.W, dim=1)
            s = (c * u_hat).sum(dim=1, keepdim=True)
            v = self._squash(s)

            if _ < self.iterations - 1:
                u_hat = u + (v * (u * s).sum(dim=1, keepdim=True))

        # 输出
        return v.squeeze(dim=1)

    def _squash(self, x, dim=-1):
        norm = x.norm(dim=dim, keepdim=True)
        scale = norm ** 2 / (1 + norm ** 2)
        return scale * x / norm








