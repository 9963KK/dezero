# 线性回归
import dezero.functions as F
from dezero import Variable
import numpy as np
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
# import matplotlib.pyplot as plt
# plt.scatter(x, y)
# plt.show()
x, y = Variable(x), Variable(y)
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))
def predict(x):
    y = F.matmul(x, W) + b
    return y
# 当x.shape[1] == W.shape[0]时，就能正确计算。
def mean_suqared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

lr = 0.1
iters = 100
for i in range(iters):
    y_pred = predict(x)
    loss = mean_suqared_error(y, y_pred)
    W.cleargrad()
    b.cleargrad()
    loss.backward()
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(W, b, loss)