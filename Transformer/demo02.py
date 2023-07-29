from sympy import symbols, diff, integrate

# 定义符号变量
x = symbols('x')
# 定义被积函数
f = x**2 + 2*x + 1
# 对函数进行微分
df = diff(f, x)
# 对微分后的函数进行积分
result = integrate(df, x)
# 打印结果
print("积分结果:", result)