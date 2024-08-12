# 实现高阶导数
# 泰勒展开的导数
import numpy as np
from dezero import Function
from dezero import Variable
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx
def sin(x):
    return Sin()(x)

# 实现泰勒展开
# 根据泰勒展开公式，展开的项数越多，导数的精度越高
import math
# 使用factorial函数
def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(10000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

# x = Variable(np.array(np.pi / 4))
# y = my_sin(x)
# y.backward()
# print(y.data)
# print(x.grad)
# 误差较小，因为这个x = pi / 4是 cos(x) = sin(x)因此计算出来的grad和值应该是相同的。