# 汇总层的层
import dezero.layers as L
import dezero.functions as F
from dezero import Variable, Layer, Model

model = Layer()
model.l1 = L.Linear(5)
model.l2 = L.Linear(3)

def predict(x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y
for p in model.params():
    print()
model.cleargrads()