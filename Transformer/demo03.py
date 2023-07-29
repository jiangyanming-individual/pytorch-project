from sympy import symbols, diff, integrate
from scipy.integrate import quad
import numpy as np



def funcU(x,t):

    U=x ** 2 +t**2
    return U


# 定义符号变量
x = symbols('x')

t=np.random.uniform(0,1,20) #时间

# 定义被积函数
df = (T - t) * (-r) * du

u=funcU(x,t)

# 对函数进行微分
du = diff(u, t)

# 对微分后的函数进行积分
result = integrate(df,t)
# 打印结果
print("积分结果:", result)