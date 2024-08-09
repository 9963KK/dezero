# 减少内存用量
import numpy as np
import weakref
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported.".format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generations = 0
    def set_creator(self, func):
        self.creator = func
        self.generations = func.generations + 1
    def cleargrad(self):
        self.grad = None
    def backward(self,retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generations)
        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
                # 保证下面的zip能正常运行
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y是weakref
class Config:
    enable_backprop = True
# 使用反向传播和禁用反向传播的模式切换
# 使用with函数来切换enable_backprop

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    def forward(self, xs):
        raise NotImplementedError
    def backward(self, gys):
        raise NotImplementedError
class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    def backward(self, gys):
        x = self.inputs[0].data
        gx = 2 * x * gys
        return gx
def square(x):
    return Square()(x)
# 不保留不必要的导数
# 在Variable的backward中添加是否保留导数的参数

# 引入contextlib
import contextlib

@contextlib.contextmanager
# def config_test():
#     print("start")
#     try:
#         yield
#     finally:
#         print("end")
# with config_test():
#     print("process")
    # with处理可能会出现异常，会被发送到yield语句中，即需要使用try-finally
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = square(x)