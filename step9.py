# 优化函数

import numpy as np


class Variable:
    def __init__(self, data):
        if not data:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x = f.input
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):
        y = self.forward(input.data)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Square(Function):
    def forward(self, input: Variable):
        self.x = input
        y = self.x ** 2
        return y

    def backward(self, gy):
        gx = 2 * self.x * gy
        return gx


class Exp(Function):
    def forward(self, input):
        self.x = input
        y = np.exp(self.x)
        return y

    def backward(self, gy):
        gx = np.exp(self.x) * gy
        return gx

# 1. 把函数的调用步骤优化
def square(x):
    return Square()(x)
def exp(x):
    return Exp()(x)

# 2.简化backward方法，可以省略y.grad = np.arrary(1.0)
# 往backward里面加入初始化方法
# if self.grad is None:
#       self.grad = np.ones_like(self.data)

# 3. 对非ndarray进行报错。
# 4. 对输出的标量进行转换，转为ndarray
def as_array(data):
    if np.isscalar(data):
        return np.asarray(data)
    return data

x = Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)