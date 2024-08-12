# 函数优化
# 对于Rosenbrock函数进行优化，找到最小点

#1. 求导
import numpy as np
from dezero import Variable

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

# 查看函数在(0, 2)点的取值。
x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
# y = rosenbrock(x0, x1)
# y.backward()
# print(x0.grad, x1.grad)
# -x0.grad, x1.grad 是下降最快的方向
# 接下来进行迭代

lr = 0.001
iters = 1000
for i in range(iters):
    print(x0, x1)
    y = rosenbrock(x0, x1)
    
    x0.cleargrad()
    x1.cleargrad()
    y.backward()
    
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad