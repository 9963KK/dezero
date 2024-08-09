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

    def backward(self, gys):
        raise NotImplementedError

# 实现加法类


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
# 对调用时的方法进行改进
# 结果不再需要调用元组返回
# x0 = Variable(np.array(2))
# x1 = Variable(np.array(3))
# f = Add()
# y = f(x0, x1)
# print(y.data)

# 对Function的方法也进行改写，使用*直接解包xs
def add(x0, x1):
    return Add()(x0, x1)