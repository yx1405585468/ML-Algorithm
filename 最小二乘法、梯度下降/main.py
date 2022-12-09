"""
    线性回归的梯度最小二乘法求解，以及梯度下降策略
"""
import numpy as np
from matplotlib import pyplot as plt


# 随机梯度下降：定义不同时间的学习率，时间越靠后，学习率越小，刚开始，学习率较大
def learning_schedule_sgd(t_):  # 随机梯度下降的学习率更新
    t0 = 5
    t1 = 50
    return t0 / (t1 + t_)


def learning_schedule_minibatch(t_):  # 小批量梯度下降的学习率更新
    t0, t1 = 200, 1000
    return t0 / (t_ + t1)


if __name__ == '__main__':
    # TODO 1. 制造数据，可视化数据
    x = 2 * np.random.rand(100, 1)  # 100行 1列
    y = 4 + 3 * x + np.random.rand(100, 1)  # 100行 1列

    # TODO 2. 最小二乘法直接求解最佳参数
    x_1 = np.c_[np.ones((100, 1)), x]  # np.c_：连接2个矩阵，加上1列 截距
    theta_best = np.linalg.inv(x_1.T.dot(x_1)).dot(x_1.T).dot(y)  # np.linalg 线性代数的函数，inv是求矩阵的逆

    # TODO 2.1 制造数据验证参数预测
    x_new = np.array([[0], [2]])
    x_new_1 = np.c_[np.ones((2, 1)), x_new]  # np.c_：连接2个矩阵，加上1列 截距
    y_predict = x_new_1.dot(theta_best)
    print("最小二乘法求解的参数预测结果：\n", y_predict)

    # TODO 3.不同梯度下降策略

    # TODO 3.1 批量梯度下降
    theta_path_bgd = []
    learning_rate = 0.1
    n_iterations = 1000
    m = 100
    theta = np.random.rand(2, 1)
    for i in range(n_iterations):
        gradients = 2 / m * x_1.T.dot(x_1.dot(theta) - y)
        theta = theta - learning_rate * gradients
        theta_path_bgd.append(theta)
    y_predict = x_new_1.dot(theta)
    print("批量梯度下降求解的参数预测结果：\n", y_predict)

    # TODO 3.2 SGD随机梯度下降
    theta_path_sgd = []  # 梯度下降的前进路线
    m = len(x_1)  # 总数据量
    n_epochs = 50  # 整个样本批训练50次

    theta = np.random.rand(2, 1)
    for epoch in range(n_epochs):  # 全部数据集训练50个批次
        for i in range(m):  # 在每一个批次的数据集训练中，遍历每一个数据点
            if epoch < 10 and i < 10:  # 记录前几步的梯度前进路线，全部记录就太乱了，也可以尝试一下
                y_predict = x_new_1.dot(theta)
                plt.plot(x_new, y_predict, "r-")
            random_index = np.random.randint(m)  # 因为是随机梯度下降，所以要随机选择一个样本，这是随机样本的index
            xi = x_1[random_index:random_index + 1]  # 找到随机的这个点的值
            yi = y[random_index:random_index + 1]  # 找到随机的这个点对应的标签
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)  # 计算这个点的梯度
            eta = learning_schedule_sgd(epoch * m + i)  # 根据当前进行的批次和迭代的位置计算当前的学习率
            theta = theta - eta * gradients  # 往梯度下降的方向进行迭代，更新参数
            theta_path_sgd.append(theta)  # 记录梯度更新的前行路径
    plt.plot(x, y, "b.")
    plt.axis([0, 2, 0, 15])

    # TODO 3.2 MiniBatch梯度下降
    theta_path_mgd = []
    n_epochs = 50
    minibatch = 16
    theta = np.random.randn(2, 1)

    t = 0
    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(m)
        x_1_shuffled = x_1[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, minibatch):
            t += 1
            xi = x_1_shuffled[i:i + minibatch]
            yi = y_shuffled[i:i + minibatch]
            gradients = 2 / minibatch * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule_minibatch(t)
            theta = theta - eta * gradients
            theta_path_mgd.append(theta)

    # TODO 4 比较三种梯度下降策略的好坏
    theta_path_bgd = np.array(theta_path_bgd)
    theta_path_sgd = np.array(theta_path_sgd)
    theta_path_mgd = np.array(theta_path_mgd)
    plt.figure(figsize=(12, 6))
    plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], 'r-s', linewidth=1, label='SGD')
    plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], 'g-+', linewidth=2, label='MINIGD')
    plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], 'b-o', linewidth=3, label='BGD')
    plt.legend(loc='upper left')
    plt.axis([3.5, 4.5, 2.0, 4.0])
    plt.show()

    # TODO 5 正则化解决过拟合
    # 岭回归
    # lasso回归
