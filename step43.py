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
# Input, Hidden, Output表示三个不同层的维度
# 表示形状
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

# 2️⃣ 神经网络实现
def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2
iters = 10000

# 3️⃣ 神经网络寻的训练
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:
        print(loss)
# 这个过程相当于从loss函数出发，一步步backward，并更新权重
import matplotlib.pyplot as plt
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01).reshape(-1, 1)
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r')
plt.show()

