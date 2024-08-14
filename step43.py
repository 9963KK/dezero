# 神经网络
# 实现非线性数据集的计算

import numpy as np
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
