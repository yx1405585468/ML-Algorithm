import time

from sklearn.datasets import load_iris
from 逻辑回归.utils.features import prepare_for_training
from 逻辑回归.utils.hypothesis import sigmoid
import numpy as np


class LogisticRegression:
    def __init__(self, alpha=0.01, num_iterations=1000):
        self.data = None
        self.label = None
        self.label_unique = None

        self.num_data = None
        self.num_label_unique = None
        self.num_features = None

        self.theta = None

        self.alpha = alpha
        self.num_iterations = num_iterations

    def fit(self, data, label):
        # 1. 初始化数据
        self.data = prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=False)[0]
        self.label = label
        self.label_unique = np.unique(self.label)  # [0 1 2]

        # 2. 统计数据量、标签类别量、特征量
        self.num_data = self.data.shape[0]
        self.num_label_unique = self.label_unique.shape[0]
        self.num_features = self.data.shape[1]

        # 3. 初始化参数θ，θ.T的维度是(特征量，标签类别)，所以θ维度是(标签类别，特征量)
        self.theta = np.zeros((self.num_label_unique, self.num_features))

        # 4. 根据迭代次数进行一个梯度迭代
        for index, label in enumerate(self.label_unique):
            # 从所有初始化参数中，选取当前label的初始化参数,并reshape，至θ*data
            current_theta = np.copy(self.theta[index].reshape(self.num_features, 1))

            # 再将多分类转化为2分类，在所有label中，等于当前label的为1，不等于当前label的为0
            current_label = np.array((self.label == label).astype(float)).reshape(-1, 1)

            # 对当前二分类执行梯度下降，求最终的参数，还有损失函数的历史
            current_theta = self.gradient_step(current_label, current_theta)

    def gradient_step(self, current_label, current_theta):
        for i in range(self.num_iterations):
            prediction = sigmoid(np.dot(self.data, current_theta))
            error = prediction - current_label
            gradient = (1 / self.num_data) * np.dot(self.data.T, error)

            current_theta = current_theta - self.alpha * gradient

        return current_theta


if __name__ == '__main__':
    # 1. 制造数据
    data_ = load_iris().data
    label_ = load_iris().target

    # 2. 训练预测
    lgr = LogisticRegression()
    lgr.fit(data_, label_)
