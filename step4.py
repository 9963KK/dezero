# 导入微分的方法
from step1 import Variable, Function, Square
from step3 import Exp
import numpy as np

# 实现数值微分
def num_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
# 测试功能
# x = Variable(np.array(2.0))
# y = num_diff(Square(), x)
# print(y)

# 复合函数求导
x = Variable(np.array(0.5))
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))
dy = num_diff(f, x)
print(dy)

# 但是这种计算存在精度丢失的问题，即在计算过程中的有效位被丢弃。
# 同时在数据量巨大时，这个方法也是不合理的，过于复杂。
