# ===============================
# 1. 导入必要的库
# ===============================

import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 2. 生成模拟数据（真实世界 + 噪声）
# ===============================

np.random.seed(42)  # 固定随机种子，保证每次结果一致（实验可复现）

n = 50  # 样本数量

# 自变量 x：从 0 到 10 均匀采样
x = np.random.uniform(0, 10, size=n)

# 噪声 epsilon：服从均值为 0，标准差为 2 的高斯分布
epsilon = np.random.normal(0, 2, size=n)

# 真实线性关系
# y = 3x + 5 + noise
y = 3 * x + 5 + epsilon

# ===============================
# 3. 可视化原始数据
# ===============================

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Observed Data")
plt.show()

# ===============================
# 4. 构造线性回归模型（矩阵形式）
# ===============================

# 我们的模型是：
# y = b + w*x
#
# 写成矩阵形式：
# y = X @ beta
#
# X = [1, x]
# beta = [b, w]^T

# 构造设计矩阵 X（n 行 2 列）
X = np.column_stack([
    np.ones(n),  # 第一列全 1，对应截距 b
    x            # 第二列是 x，对应斜率 w
])

# 把 y 变成列向量（n x 1）
y_vec = y.reshape(-1, 1)

# ===============================
# 5. 使用正规方程求最小二乘解
# ===============================

# 数学公式：
# beta_hat = (X^T X)^(-1) X^T y

XT_X = X.T @ X              # X^T X
XT_X_inv = np.linalg.inv(XT_X)  # (X^T X)^(-1)
XT_y = X.T @ y_vec          # X^T y

beta_hat = XT_X_inv @ XT_y  # 最终解

# 拆出参数
b_hat = beta_hat[0, 0]  # 截距
w_hat = beta_hat[1, 0]  # 斜率

print("Estimated parameters:")
print("w =", w_hat)
print("b =", b_hat)

# ===============================
# 6. 可视化拟合结果
# ===============================

# 用学到的模型画一条直线
x_line = np.linspace(0, 10, 100)
y_line = w_hat * x_line + b_hat

plt.scatter(x, y, label="Data")
plt.plot(x_line, y_line, label="Fitted Line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Fit")
plt.show()

# ===============================
# 7. 残差分析（模型诊断）
# ===============================

# 预测值
y_pred = X @ beta_hat  # X beta

# 残差 = 真实值 - 预测值
residuals = y_vec - y_pred

plt.scatter(x, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("x")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()

# ===============================
# 8. 计算误差指标（MSE）
# ===============================

# 均方误差：最小二乘真正最小化的量
mse = np.mean(residuals ** 2)
print("Mean Squared Error:", mse)