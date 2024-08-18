# 构建softmax函数
from dezero.models import MLP
import numpy as np
model = MLP((10, 3))
x = np.array([[0.2, -0.4]])
y = model(x)
print(y)