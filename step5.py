# 对反向传播方法的实现
# 实际上就是把每一个步骤的导数相乘
# 扩展Variable类，加入梯度变量
import numpy as np
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None # 反向传播时设置其为计算出的导数值
        
# Function的扩展：
# 1. 记录每次input的值
# 2. 加入backward的运算
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        self.input = input # 保存输入变量
        output = Variable(y)
        return output
    def forward(self, x):
        raise NotImplementedError
    def backward(self, gy):
        raise NotImplementedError
# 下面就是对Square和Exp函数的重构
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


# 这里面的的gy这个变量相当于在正向计算中输出端反向传回的导数。
A = Square()
B = Exp()
C = Square()
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)