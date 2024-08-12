# DeZero的用途

import numpy as np
from dezero import Variable

# double backprop
x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph=True)

gx = x.grad
x.cleargrad()

z = gx ** 3 + y
z.backward()
print(x.grad)

# 牛顿法之所以很少用到神经网络中是因为当输入的参数变大时，计算复杂度会变的很大（黑塞矩阵），不是一个现实的方案