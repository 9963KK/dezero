# 汇总层的层
import dezero.layers as L
import dezero.functions as F
from dezero import Variable, Model
import numpy as np

# model = Layer()
# model.l1 = L.Linear(5)
# model.l2 = L.Linear(3)

# def predict(x):
#     y = model.l1(x)
#     y = F.sigmoid(y)
#     y = model.l2(y)
#     return y
# for p in model.params():
#     print()
# model.cleargrads()

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y
model = TwoLayerNet(hidden_size, 1)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y_pred, y)
    
    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)