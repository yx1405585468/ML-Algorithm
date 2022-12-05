from sklearn.datasets import load_iris
from 逻辑回归.utils.features import prepare_for_training
from 逻辑回归.utils.hypothesis import sigmoid
from sklearn.metrics import accuracy_score
import numpy as np


class LogisticRegression:
    def __init__(self, alpha=0.1, num_iterations=1000):
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

            # 对当前二分类执行梯度下降，求当前二分类算法最终参数，
            current_theta, cost = self.gradient_step(current_label, current_theta)

            # 将当前二分类参数
            self.theta[index] = current_theta.T

    def gradient_step(self, current_label, current_theta):
        for i in range(self.num_iterations):
            prediction = sigmoid(np.dot(self.data, current_theta))  # sigmoid函数包裹，将结果映射到0-1之间
            error = prediction - current_label  # 计算错误标签
            gradient = (1 / self.num_data) * np.dot(self.data.T, error)  # 计算梯度
            current_theta = current_theta - self.alpha * gradient  # 迭代更新参数
            cost = self.cost_function(current_label, current_theta)
            print(cost)

        return current_theta, cost

    def cost_function(self, current_label, current_theta):

        predictions = sigmoid(np.dot(self.data, current_theta))
        y_is_set_cost = np.dot(current_label[current_label == 1].T, np.log(predictions[current_label == 1]))
        y_is_not_set_cost = np.dot(1 - current_label[current_label == 0].T, np.log(1 - predictions[current_label == 0]))
        cost = (-1 / self.num_data) * (y_is_set_cost + y_is_not_set_cost)
        return cost

    def predict(self, data):
        num_data = data.shape[0]
        data_processed = prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=False)[0]
        prob = sigmoid(np.dot(data_processed, self.theta.T))
        # print(prob)  # 预测出的结果分别是三类中每一类的概率
        # [[9.94278388e-01 1.32253688e-01 1.17526077e-06]
        #  [9.84413654e-01 2.48847537e-01 4.58543908e-06]
        #  [9.90857054e-01 1.62324192e-01 3.52917292e-06]]
        max_prob_index = np.argmax(prob, axis=1)  # 找到概率最大的一列的索引
        class_prediction = np.empty(max_prob_index.shape, dtype=object)  # 根据索引的shape制造一列列表
        for index, label in enumerate(self.label_unique):
            class_prediction[max_prob_index == index] = label  # 以预测索引对应label的索引，进而匹配出预测的标签label
        return class_prediction.reshape((num_data, 1))


if __name__ == '__main__':
    # 1. 制造数据
    data_ = load_iris().data
    label_ = load_iris().target

    # 2. 训练预测
    lgr = LogisticRegression()
    lgr.fit(data_, label_)
    result = lgr.predict(data_).reshape(1, -1)[0].tolist()
    print("准确率", accuracy_score(label_.tolist(), result))
