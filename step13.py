# 反向传播的可变长参数优化

import numpy as np
# 把step12里面的Variable和Function都拿过来
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
class Variable:
    def __init__(self, data):
        if not data:
            if not np.isscalar(data):
                raise TypeError("{} is not supported.".format(type(data)))
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
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx
                if x.creator is not None:
                    funcs.append(x.creator)
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]
    def forward(self, xs):
        raise NotImplementedError
    def backward(self, gy):
        raise NotImplementedError
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    def backward(self, gy):
        return gy, gy
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
def square(x):
    return Square()(x)
x1 = Variable(np.array(2.0))
x2 = Variable(np.array(3.0))
f = Add()
y = f(square(x1), square(x2))
y.backward()
print(y.data)
print(x1.grad)
print(x2.grad)