# 对构建的类进行测试
import unittest
from step9 import *
class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
    def test_backward_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(num_grad, x.grad)
        self.assertTrue(flg)

# 使用梯度检验来进行自动测试
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data + eps)
    x1 = Variable(x.data - eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y0.data - y1.data) / (2 * eps)

unittest.main()

# 自动微分其实是前向和后向的集合，计算机求导一般为三种：
# 1️⃣ 数值微分
# 2️⃣ 符号微分
# 3️⃣ 自动微分