"""
最小二乘法（线性回归）“教会版”

你说的对：只跑代码并不会真的学会。这个文件做 4 件事：
1) 直观解释“最小二乘到底在最小什么”
2) 推导正规方程（X^T X)^(-1) X^T y 的来历
3) 逐行解释 + 关键变量打印
4) 给你一个“手算小练习”
"""

# ===============================
# 0. 可调参数
# ===============================
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # 固定随机种子，保证每次结果一致（实验可复现）
SHOW_DEBUG = True  # 想看关键中间量就打开
SHOW_PLOTS = True  # 不想弹窗就关掉

# ===============================
# 1. 直观理解：最小二乘到底在最小什么
# ===============================
# 给定数据点 (x_i, y_i)，我们用一条直线 y = b + w*x 去拟合。
# 每个点都有一个“竖直距离”（残差）:
#   r_i = y_i - (b + w*x_i)
# 最小二乘就是把所有残差的平方求和，作为损失：
#   L(b, w) = sum_i (y_i - (b + w*x_i))^2
# 为什么要平方？因为：
# - 绝对值不可导点多，不方便求解
# - 平方能放大大误差、让优化更稳定
#
# 所以“最小二乘”本质：选一条线，让所有点到这条线的竖直距离的平方总和最小。

# ===============================
# 2. 生成模拟数据（真实世界 + 噪声）
# ===============================
n = 50  # 样本数量
x = np.random.uniform(0, 10, size=n)        # 自变量 x：0 到 10 均匀采样
epsilon = np.random.normal(0, 2, size=n)    # 噪声：均值 0，标准差 2
y = 3 * x + 5 + epsilon                     # 真实线性关系：y = 3x + 5 + noise

if SHOW_PLOTS:
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Observed Data")
    plt.show()

# ===============================
# 3. 矩阵形式：把问题写成 y = X @ beta
# ===============================
# 模型：y = b + w*x
# 写成矩阵：
#   y = X @ beta
#   X = [1, x]  (n x 2)
#   beta = [b, w]^T

X = np.column_stack([
    np.ones(n),  # 第一列全 1，对应截距 b
    x            # 第二列是 x，对应斜率 w
])
y_vec = y.reshape(-1, 1)  # (n x 1)

if SHOW_DEBUG:
    print("\n[DEBUG] X shape:", X.shape)
    print("[DEBUG] y_vec shape:", y_vec.shape)

# ===============================
# 4. 正规方程推导（核心理解）
# ===============================
# 我们的损失：
#   L(beta) = || y - X beta ||^2
# 展开：
#   L = (y - Xb)^T (y - Xb)
# 对 beta 求导，设为 0：
#   dL/dbeta = -2 X^T (y - Xb) = 0
#   => X^T y - X^T X b = 0
#   => (X^T X) b = X^T y
#   => b = (X^T X)^(-1) X^T y
# 这就是“正规方程”。

# ===============================
# 5. 使用正规方程求最小二乘解
# ===============================
XT_X = X.T @ X                  # X^T X
XT_y = X.T @ y_vec              # X^T y
XT_X_inv = np.linalg.inv(XT_X)  # (X^T X)^(-1)
beta_hat = XT_X_inv @ XT_y      # 最终解

b_hat = beta_hat[0, 0]
w_hat = beta_hat[1, 0]

print("\nEstimated parameters:")
print("w =", w_hat)
print("b =", b_hat)

if SHOW_DEBUG:
    print("\n[DEBUG] X^T X:\n", XT_X)
    print("[DEBUG] (X^T X)^-1:\n", XT_X_inv)
    print("[DEBUG] X^T y:\n", XT_y)
    print("[DEBUG] beta_hat:\n", beta_hat)

# ===============================
# 6. 可视化拟合结果
# ===============================
x_line = np.linspace(0, 10, 100)
y_line = w_hat * x_line + b_hat

if SHOW_PLOTS:
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
y_pred = X @ beta_hat
residuals = y_vec - y_pred

if SHOW_PLOTS:
    plt.scatter(x, residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("x")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.show()

# ===============================
# 8. 误差指标（MSE）
# ===============================
mse = np.mean(residuals ** 2)
print("Mean Squared Error:", mse)

# ===============================
# 9. 手算练习（真正学会的关键）
# ===============================
# 用 3 个点手算一遍：
# (x, y) = (1, 2), (2, 3), (3, 5)
# 1) 写出 X 和 y
# 2) 计算 X^T X、X^T y
# 3) 手算 beta_hat = (X^T X)^-1 X^T y
# 4) 对比代码结果
#
# 你可以把下面的 block 打开来验证手算是否一致：
#
# x_small = np.array([1.0, 2.0, 3.0])
# y_small = np.array([2.0, 3.0, 5.0])
# X_small = np.column_stack([np.ones_like(x_small), x_small])
# y_small_vec = y_small.reshape(-1, 1)
# beta_small = np.linalg.inv(X_small.T @ X_small) @ (X_small.T @ y_small_vec)
# print("\n[EXERCISE CHECK] beta_small =", beta_small.ravel())
