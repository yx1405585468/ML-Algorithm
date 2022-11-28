import numpy as np
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # TODO 1 正则化解决过拟合

    np.random.seed(42)
    m = 20
    x = 3 * np.random.rand(m, 1)
    y = 0.5 * x + np.random.randn(m, 1) / 1.5 + 1
    x_new = np.linspace(0, 3, 100).reshape(100, 1)
