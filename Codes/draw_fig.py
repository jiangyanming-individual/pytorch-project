# author:Lenovo
# datetime:2023/7/16 19:21
# software: PyCharm
# project:pytorch项目


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
#--------------------------------------------------
import torch
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 从 -1 到 1 选取 10 个点
x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)

# 生成所有可能的点对，并计算其范数
points = []
for i in range(10):
    for j in range(10):
        point = np.array([x[i], y[j]])
        norm = np.linalg.norm(point)
        if norm < 1:
            points.append(point)

# 将小于 1 的点按照 x 坐标排序
points = sorted(points, key=lambda p: p[0])

# 提取 x 坐标和 y 坐标，分别转换为张量
pt_x = torch.tensor([p[0] for p in points]).reshape(-1, 1)
pt_y = torch.tensor([p[1] for p in points]).reshape(-1, 1)


pt_output=(1-(torch.square(pt_x)+ torch.square(pt_y))) ** 1.75


new_pt_output=torch.squeeze(pt_output).detach().numpy()
for i in new_pt_output:
    print(i)

pt_output=pt_output.detach().numpy()
# print(pt_output)


# #将函数值转换为 NumPy 数组，并使用 imshow 函数可视化
# pt_f = pt_output.detach().numpy()
# print("pt_f:",pt_f)
# print("pt_f shape",pt_f.shape)
#
# plt.imshow(pt_f, cmap='hot', interpolation='nearest')
# plt.colorbar()

# plt.show()
#
#
# #
# #
# #------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_zlim([0,1])
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])

# ax.invert_zaxis()
ax.invert_xaxis()

ax.set_xticks(np.arange(-1,1.2,0.2))
ax.set_yticks(np.arange(-1,1.2,0.2))
ax.set_zticks(np.arange(0,1.5,0.5))


pt_x=pt_x.squeeze().detach().numpy()
pt_y=pt_y.squeeze().detach().numpy()

ax.plot_surface(pt_x, pt_y, pt_output, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.003, antialiased=True)

ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')

ax.xaxis.pane.fill = False#面板不填充颜色
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

plt.show()