# 正向传播部分
# 可变长参数
# 修改Function使其能够容纳多个参数
import numpy as np
def as_array(x):
    if np.isscalar(x):
        return np.asarray(x)
    return x
class Variable:
    def __init__(self, data):
        if not data:
            if not np.isinstance(data, np.ndarray):
                raise TypeError("{} is not supported.".format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
    def set_creator(self, func):
        self.creator = func
    def backward(self):
        if not self.grad:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)
class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs
    def forward(self, xs):
        raise NotImplementedError
    def backward(self, gys):
        raise NotImplementedError

# 实现加法类
class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)
