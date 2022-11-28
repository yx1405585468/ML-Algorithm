import numpy as np
from matplotlib import pyplot as plt


# 随机梯度下降：定义不同时间的学习率，时间越靠后，学习率越小，刚开始，学习率较大
def learning_schedule(t):
    t0 = 5
    t1 = 50
    return t0 / (t1 + t)


if __name__ == '__main__':
    # TODO 1. 制造数据，可视化数据
    x = 2 * np.random.rand(100, 1)  # 100行 1列
    y = 4 + 3 * x + np.random.rand(100, 1)  # 100行 1列

    # TODO 2. 最小二乘法直接求解最佳参数
    x_1 = np.c_[np.ones((100, 1)), x]  # np.c_：连接2个矩阵，加上1列 截距
    theta_best = np.linalg.inv(x_1.T.dot(x_1)).dot(x_1.T).dot(y)  # np.linalg 线性代数的函数，inv是求矩阵的逆

    # 2.1 制造数据验证参数预测
    x_new = np.array([[0], [2]])
    x_new_1 = np.c_[np.ones((2, 1)), x_new]  # np.c_：连接2个矩阵，加上1列 截距
    y_predict = x_new_1.dot(theta_best)
    print("最小二乘法求解的参数预测结果：\n", y_predict)

    # TODO 3.不同梯度下降策略造成的影响

    # 3.1 批量梯度下降
    learning_rate = 0.1
    n_iterations = 1000
    m = 100
    theta = np.random.rand(2, 1)
    for i in range(n_iterations):
        gradients = 2 / m * x_1.T.dot(x_1.dot(theta) - y)
        theta = theta - learning_rate * gradients
    y_predict = x_new_1.dot(theta)
    print("批量梯度下降求解的参数预测结果：\n", y_predict)

    # 3.2 SGD随机梯度下降
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
            eta = learning_schedule(epoch * m + i)  # 根据当前进行的批次和迭代的位置计算当前的学习率
            theta = theta - eta * gradients  # 往梯度下降的方向进行迭代，更新参数
            theta_path_sgd.append(theta)  # 记录梯度更新的前行路径
    plt.plot(x, y, "b.")
    plt.axis([0, 2, 0, 15])
    plt.show()
