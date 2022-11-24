import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from 线性回归.utils.features import prepare_for_training


class LinearRegression:
    def __init__(self, alpha=0.01, num_iterations=500):
        """
        0. 模拟sklearn框架，fit and predict
        1. 初始化学习率和梯度的迭代次数，alpha=0.01, num_iterations=500
        2. data和label初始为None
        3. 参数初始为None，特征列数，初始为None

        """
        # 1. 训练数据集和标签
        self.data = None
        self.label = None

        # 2. 矩阵参数和特征列数
        self.theta = None
        self.num_features = None

        # 3. 迭代次数和学习率
        self.num_iterations = num_iterations
        self.alpha = alpha

    def fit(self, data, label):
        """
                定义fit模块
        """
        # 1. 初始化
        self.data = data
        self.label = label

        # 2. 赋值矩阵参数和特征列数
        self.num_features = self.data.shape[1]
        self.theta = np.zeros((self.num_features, 1))

        # 3. 根据迭代次数和学习率，迭代进行梯度下降
        for _ in range(self.num_iterations):
            self.gradient_step(self.alpha)
            self.cost_function()

    def predict(self, data):
        """
                定义predict模块
        """
        prediction = self.predict_inner(data)
        return prediction

    def predict_inner(self, data):
        """
                定义predict模块
        """
        # 1. data 矩阵相乘θ 即为预测结果
        prediction = np.dot(data, self.theta)
        return prediction

    def gradient_step(self, alpha):
        """
                核心模块！！！ 梯度下降参数更新计算方法，注意是矩阵运算
                链接：https://www.bilibili.com/video/BV1US4y1v7Pd?p=12&vd_source=911355ef9c6b4864d6d46e451678e78d
        """
        # TODO：!!!核心公式：小批量梯度下降，根据学习率，更新θ参数
        num_examples = self.data.shape[0]  # 获取数据的行数
        prediction = self.predict_inner(self.data)  # 每迭代一次的预测值
        error = prediction - self.label  # 计算每一次迭代的误差
        self.theta = self.theta - alpha * (1 / num_examples) * (np.dot(error.T, self.data)).T

    def cost_function(self):
        """
                    损失计算方法
        """
        num_examples = self.data.shape[0]
        delta = self.predict_inner(self.data) - self.label
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples
        print(cost[0][0])


if __name__ == '__main__':
    # 1. 原始数据集
    data = pd.read_csv('../../../线性回归/data/non-linear-regression-x-y.csv')
    x = data['x'].values.reshape((data.shape[0], 1))
    y = data['y'].values.reshape((data.shape[0], 1))

    # 2. 预处理，标准化，并且加一列1
    x_precessed = prepare_for_training(x, polynomial_degree=15, sinusoid_degree=15, normalize_data=True)[0]

    # 3. 训练
    linear = LinearRegression(alpha=0.02, num_iterations=50000)
    linear.fit(x_precessed, y)

    # 4. 预测
    pre_result = linear.predict(x_precessed)

    # 5. 可视化
    plt.plot(x, pre_result)
    plt.show()
