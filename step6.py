# 实现反向传播自动化
# 向Variable加入creator，表示函数值由函数创造。
import numpy as np
# class Variable:
#     def __init__(self, data):
#         self.data = data
#         self.grad = None
#         self.creator = None
#     def set_creator(self, func):
#         self.creator = func
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output
    def forward(self, x):
        raise NotImplementedError
    def backward(self, gy):
        raise NotImplementedError
class Square(Function):
    def forward(self, x):
        return x ** 2
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# assert y.creator == C
# assert y.creator.input == b
# assert y.creator.input.creator == B
# assert y.creator.input.creator.input == a
# assert y.creator.input.creator.input.creator == A
# assert y.creator.input.creator.input.creator.input == x
# y.grad = np.array(1.0)
# C = y.creator
# b = C.input
# b.grad = C.backward(y.grad)
# B = b.creator
# a = B.input
# a.grad = B.backward(b.grad)
# A = a.creator
# x = A.input
# x.grad = A.backward(a.grad)
# print(x.grad)

# 往变量里添加反向传播的方法
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    def set_creator(self, func):
        self.creator = func
    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()


x = Variable(np.array(0.5))
A = Square()
B = Exp()
C = Square()
a = A(x)
b = B(a)
y = C(b)
y.grad = np.array(1.0)
y.backward()
print(x.grad)
