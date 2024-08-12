# sin高阶导数

import numpy as np
import dezero.functions as F
from dezero import Variable
import matplotlib.pyplot as plt
from dezero import Variable

# x = Variable(np.array(1.0))
# y = F.sin(x)
# y.backward(create_graph=True)

# for i in range(3):
#     gx = x.grad
#     x.cleargrad()
#     gx.backward(create_graph=True)
#     print(x.grad)

x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)
logs = [y.data]

for i in range(3):
    logs.append(x.grad.data)
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc='lower right')
plt.show()