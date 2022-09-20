import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


# target function
def real_func(x):
    return np.sin(2 * np.pi * x)


# fit function
def fit_func(p, x):
    """
    numpy.poly1d([1,2,3]) -> 1(x*x) + 2(x) + 3
    """
    f = np.poly1d(p)
    return f(x)


def residuals_func(p, x, y):
    res = fit_func(p, x) - y
    return res


def leastsq_mutifunc(x, y, m):
    """
    多项式最小二乘法实现
    :param x:输入
    :param y:目标输出
    :param m:多项式阶数
    :return:多项式系数
    """
    x = np.array(x)
    y = np.array(y)

    assert m <= x.shape[0], f"the number of m({m}) need less than x's size({x.shape[0]})"
    assert x.shape[0] == y.shape[0], f"the size of x({x.shape[0]}) must equal to y's size({y.shape[0]}"
    x_mat = np.zeros((x.shape[0], m + 1))
    for i in range(x.shape[0]):
        x_mat_h = np.zeros((1, m + 1))
        for j in range(m + 1):
            x_mat_h[0][j] = x[i] ** (m - j)
        x_mat[i] = x_mat_h
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x_mat.T, x_mat)), x_mat.T), y.T)
    return theta


def fitting(x, y, M=0):
    """
    :param M: 拟合目标函数的多项式次数
    :return: 多项式系数
    """
    # 初始化多项式系数
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    my_p_lsq = leastsq_mutifunc(x, y, m=M)
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    print(f'fitting parameters: {p_lsq[0]}, my fitting parameters: {my_p_lsq}')

    # 可视化
    x_points = np.linspace(np.min(x), np.max(x), int(x.shape[0] * 10000))
    plt.plot(x_points, real_func(x_points), label='real curve')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x_points, fit_func(my_p_lsq, x_points), label='my fitted curve')
    plt.plot(x, y, 'bo', label='noise curve')
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    x = np.linspace(0, 1, 10)
    y_ = real_func(x)
    # noise
    y = [np.random.normal(0, 0.1) + y1 for y1 in y_]

    p_lsq_0 = fitting(x, y, M=9)
    # theta = leastsq_mutifunc(x, y, 4)
    # print(theta)
