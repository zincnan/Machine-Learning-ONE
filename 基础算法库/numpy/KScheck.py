from scipy import stats
import numpy as np

x = np.random.uniform(0, 10, size=1000)

# 归一化到 [0,1] 再检验标准均匀分布
x_scaled = (x - 0) / (10 - 0)

stat, p_value = stats.kstest(x_scaled, 'uniform')

print(stat, p_value)