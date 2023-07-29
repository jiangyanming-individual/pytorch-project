import matplotlib.pyplot as plt
import numpy as np


labels=['wheat','tabacum','rice','peanut','papaya','total']


five_kfold=[0.8665,0.82,0.80,0.82,0.85,0.88]
independent_test=[0.82,0.83,0.81,0.85,0.87,0.8897]


x = np.arange(1,2 *len(labels)+1,2)  # the label locations
width = 0.4  # the width of the bars
print(x)

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2,five_kfold, width, label='5kfold',color='orange',lw=1,alpha=0.8)
for i,value in zip(x,five_kfold):
    plt.text(i, value, str(value), ha='left', va='bottom',rotation=90)
rects2 = ax.bar(x + width/2,independent_test, width, label='ind_test',color='skyblue',lw=1,alpha=0.8)

for j, value in zip(x,independent_test):
    plt.text(j, value, str(value), ha='right', va='bottom',rotation=90)

# 去掉上边框和右边框
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


# 设置标题和坐标轴标签
ax.set_ylabel('AUC')
ax.set_ylim([0,1])
ax.set_title('ROC curve')
ax.set_xlim([0,12])
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 添加图例
plt.legend(loc=4)

# 显示图形
plt.show()