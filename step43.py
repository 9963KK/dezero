# 神经网络
# 实现非线性数据集的计算
import dezero.functions as F
from dezero import Variable
import numpy as np
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 神经网络通常的实现方式为linear->activation->linear->...

# 1️⃣ 权重初始化
I, H, O = 1, 10, 1
# 表示形状
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

# 2️⃣ 神经网络实现
