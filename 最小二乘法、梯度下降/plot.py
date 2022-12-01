"""
    不同学习率对梯度下降的影响
"""
import numpy as np
from matplotlib import pyplot as plt

x = 2 * np.random.rand(100, 1)  # 100行 1列
y = 4 + 3 * x + np.random.rand(100, 1)  # 100行 1列
x_1 = np.c_[np.ones((100, 1)), x]  # np.c_：连接2个矩阵，加上1列 截距
x_new = np.array([[0], [2]])
x_new_1 = np.c_[np.ones((2, 1)), x_new]  # np.c_：连接2个矩阵，加上1列 截距


def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(x_1)
    plt.plot(x, y, 'b.')
    n_iterations = 1000
    for iteration in range(n_iterations):
        y_predict = x_new_1.dot(theta)
        plt.plot(x_new, y_predict, 'b-')
        gradients = 2 / m * x_1.T.dot(x_1.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel('x_1')
    plt.axis([0, 2, 0, 15])
    plt.title('eta = {}'.format(eta))


if __name__ == '__main__':
    theta_path_bgd = []

    theta = np.random.randn(2, 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plot_gradient_descent(theta, eta=0.02)
    plt.subplot(132)
    plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
    plt.subplot(133)
    plot_gradient_descent(theta, eta=0.5)
    plt.show()
