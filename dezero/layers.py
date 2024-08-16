from dezero.core import Parameter
import weakref
import dezero.functions as F
import numpy as np

class Layer:
    def __init__(self):
        self._params = set()
    def __setattr__(self, name: str, value):
        if isinstance(value, Parameter):
            self._params.add(name)
            # 只收集Parameter参数
        super().__setattr__(name, value)
    def __call__(self, *inputs) :
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    def forward(self, inputs):
        raise NotImplementedError
    def params(self):
        for name in self._params:
            yield self.__dict__[name]
    def cleargrads(self):
        # 表明对所有参数调用cleargrad
        for param in self.params():
            param.cleargrad()

# 实现作为层的Linear
class Linear(Layer):
    def __init__(self,  out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size, self.out_size = in_size, out_size
        self.dtype = dtype
        I, O = in_size, out_size
        # W_data = np.random.randn(I, O).astype(dtype) * np.sqrt(1 / I)
        # 延迟建立输入
        # 这个初始值得设置方法是参考文献得出的
        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')
    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data
    def forward(self, x):
        # 在传播数据时初始化
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        y = F.linear(x, self.W, self.b)
        return y

