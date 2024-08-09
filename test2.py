from dezero import Variable
import numpy as np

x = Variable(np.array(1.0))
y = x + 1
y.backward()

print(y)