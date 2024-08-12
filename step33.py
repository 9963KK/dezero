# 使用牛顿法实现自动计算

import numpy as np
from dezero import Variable
def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
y = f(x)
# y.backward(create_graph=True)
# print(x.grad)
# x.cleargrad()
# gx = x.grad
# gx.backward()
# print(x.grad)

# 使用牛顿法进行优化

x = Variable(np.array(2.0))
iters = 10
for i in range(iters):
    print(i, x)
    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)
    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad
    x.data -= gx.data / gx2.data