# 实现一些特殊函数的高阶导数

# sin函数高阶导数的实现
import numpy as np
from dezero.core import Function
from dezero import utils
from dezero import Variable, as_variable, as_array
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx
class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y ** 2)
        return gx
def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx
def cos(x):
    return Cos()(x)

def tanh(x):
    return Tanh()(x)


# 实现reshape函数
from  dezero.core import as_variable

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

# 定义转置函数
class Transpose(Function):
    def forward(self, x):
        y = x.transpose()
        return y
    def backward(self, gy):
        gx = transpose(gy)
        return gx
def transpose(x):
    return Transpose()(x)

# 实现sum函数
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims) # 对gy的形状进行调整
        gx = brodcast_to(gy, self.x_shape)
        return gx
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def brodcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)
class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    def backward(self, gy):
        gx = brodcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
# 实现sum_to函数是为了实现广播功能


# 实现矩阵乘法
class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW
def matmul(x, W):
    return MatMul()(x, W)
# 反向传播世纪就是通过输入和后一项的导数输出来计算当前的导数

# 损失函数实现
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * 2 * diff / len(diff)
        gx1 = -gx0
        return gx0, gx1
def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        gx = np.exp(x) * gy
        return gx
def exp(x):
    return Exp()(x)
# 为了节省内存，见图43-1，直接在内部实现Linear
class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W) + b
        return y
    def backward(self, gy):
        x, W, b = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        if b.data is not None:
            gb = sum_to(gy, b.shape)
        else: gb = None
        return gx, gW, gb

class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None # 删除t的数据，以节省内存
    return y

def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y