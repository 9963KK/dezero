
import numpy as np
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    def set_creator(self, func):
        self.creator = func
    def backward(self):
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
        output = Variable(y)
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
A = Square()
B = Exp()
C = Square()
x = Variable(np.array(0.5))
y = A(B(C(x)))
y.grad = np.array(1.0)
y.backward()
print(x.grad)
