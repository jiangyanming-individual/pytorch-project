import matplotlib.pyplot as plt
import numpy as np


labels=['wheat','tabacum','rice','peanut','papaya','total']

ACC=[0.8224,0.8256,0.8279,0.8796,0.8512,0.8461]
MCC=[0.5079,0.4921,0.4139,0.6040,0.5285,0.5123]

independent_test=[0.8665,0.8810,0.8666,0.9216,0.8945,0.8897]


x = np.arange(1,2 *len(labels)+1,2)  # the label locations
width = 0.4  # the width of the bars
print(x)

fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2,ACC, width, label='ACC',color='orange',lw=1,alpha=0.8)
# for i,value in zip(x,ACC):
#     plt.text(i, value, str(value), ha='center', va='bottom')
#
# rects2 = ax.bar(x+width/2 ,MCC, width, label='MCC',color='skyblue',lw=1,alpha=0.8)
# for k, value in zip(x,MCC):
#     plt.text(k, value, str(value), ha='center', va='bottom')

rects3 = ax.bar(x,independent_test, width, label='AUC',color='skyblue',lw=1,alpha=0.8)
for j, value in zip(x,independent_test):
    plt.text(j, value, str(value), ha='center', va='bottom')

# 去掉上边框和右边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


# 设置标题和坐标轴标签
ax.set_ylabel('AUC')
# ax.set_ylabel('values')
ax.set_ylim([0,1])
# ax.set_title('different species')
ax.set_xlim([0,12])
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 添加图例
plt.legend(loc=4)

# 显示图形
plt.show()