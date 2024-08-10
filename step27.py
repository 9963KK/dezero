# 实现高阶导数
# 泰勒展开的导数
import numpy as np
from dezero import Function
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
