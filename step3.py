from step1 import Variable, Function, Square
import numpy as np
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
#连续调用函数
# x = Variable(np.array(0.5))
# A = Square()
# B = Exp()
# C = Square()

# a = A(x)
# b = B(a)
# y = C(b)
# print(type(y))
# print(y.data)


