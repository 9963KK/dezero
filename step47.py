# 构建softmax函数
from dezero.models import MLP
from dezero import Variable, as_variable
import dezero.functions as F
import numpy as np
model = MLP((10, 3))
x = np.array([[0.2, -0.4]])
y = model(x)
print(y)

def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x)
    sum_y = F.sum(y)
    return y / sum_y