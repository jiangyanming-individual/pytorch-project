from scipy.integrate import quad

# 定义被积函数
def f(x):
    return x**2 + 2*x + 1

# 调用quad函数计算积分
result, error = quad(f, 0, 1)  # 积分区间为[0, 1]


# 打印积分结果和误差
print("积分结果:", result)
print("估计误差:", error)

