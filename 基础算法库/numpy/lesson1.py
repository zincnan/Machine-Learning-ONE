
import numpy as np


b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(b.shape)   # (2, 3)

x = np.array([1, 2, 3])
print(x.shape)  # (3,)

x_row = x.reshape(1, 3)  # 行向量
x_col = x.reshape(3, 1)  # 列向量
print( x_row @ x_col)

A=np.zeros((3, 4))   # 3×4 全 0
B=np.ones((2, 2))    # 2×2 全 1
print(A)
print(B)