# 一个基于PyTorch实现的二分类的一维卷积胶囊网络：
# 该模型相对于前面的模型增加了一些新的组件。首先，我们可以设置重构标记reconstruction，如果该标记为True，则添加一个全连接层来重构输入。这可以提高模型的性能，并使其能够学习从输入生成输出的关系。
#
# 其次，我们可以设置use_capsule_output标志，以在模型中添加一个辅助损失层来促进胶囊向量的形成。该辅助损失层通过计算输出和目标之间的欧几里得距离来提供额外的训练信号。该辅助损失层的效果因任务而异，但在某些情况下可以显著提高模型的性能。
#
# 最后请注意，在输出中我们添加了一些额外的部分

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ConvCapsNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_capsules, capsule_dim, kernel_size, stride=1, padding=0, reconstruction=False):
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

        # 重构层
        self.reconstruction = reconstruction
        if self.reconstruction:
            self.decoder = nn.Sequential(
                nn.Linear(num_capsules * capsule_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, in_channels)
            )

        # 辅助损失层
        self.use_capsule_output = False
        if self.use_capsule_output:
            self.m_plus = 0.9
            self.m_minus = 0.1
            self.lambda_val = 0.5

            self.fc = nn.Linear(num_capsules * capsule_dim, 1)

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

        # 计算损失
        if self.use_capsule_output:
            left = F.relu(self.m_plus - self.fc(v.view(batch_size, -1)))
            right = F.relu(self.fc(v.view(batch_size, -1)) - self.m_minus)
            l_c = torch.mean(left ** 2 + self.lambda_val * right ** 2, dim=1)

        # 输出
        if self.reconstruction:
            reconstruction = self.decoder(v.view(batch_size, -1))
            return v.squeeze(dim=1), reconstruction, l_c
        else:
            if self.use_capsule_output:
                return self.fc(v.view(batch_size, -1)).sigmoid(), l_c
            else:
                return v.squeeze(dim=1)

    def _squash(self, x, dim=-1):
        norm = x.norm(dim=dim, keepdim=True)
        scale = norm ** 2 / (1 + norm ** 2)
        return scale * x / norm
