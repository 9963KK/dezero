# 首先实现Variable，实现变量的功能
class Variable:
    def __init__(self, data):
        self.data = data
        # 让Variable作为箱子使用，实际存储的数据在data里面。
import numpy as np
# data = np.array(1.0)
# x = Variable(data)
# print(x.data)

# Step 2：
# 接下来实现的Function的功能，有了Variable后就需要构建能处理Variable的Function
# class Function:
#     def __call__(self, input):
#         x = input.data
#         y = x ** 2
#         output = Variable(y)
#         return output
# 注意两点：
# 1. 函数的输入值和返回值应该是Variable
# 2. Variable实例的实际数据存储在实例变量中


# x = Variable(np.array(10))
# f = Function()
# y = f(x)
# print(type(y))
# print(y.data)

# 优化：Function作为基类来使用，有所有函数的共同功能，具体调用函数时，在继承的函数中使用。
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    def forward(self, x):
        raise NotImplementedError

# Forward方法会在继承的类中实现
class Square(Function):
    def forward(self, x):
        return x ** 2

# x = Variable(np.array(10))
# f = Square()
# y = f(x)
# print(type(y))
# print(y.data)
