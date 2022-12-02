import numpy as np
from scipy.optimize import minimize
from 逻辑回归.utils.features import prepare_for_training
from 逻辑回归.utils.hypothesis import sigmoid
from sklearn.datasets import load_iris


class LogisticRegression:
    def __init__(self, data, label, polynomial_degree=0, sinusoid_degree=0, normalize_data=False):
        data_processed, features_mean, features_deviation = prepare_for_training(data, polynomial_degree,
                                                                                 sinusoid_degree, normalize_data=False)
        self.data = data_processed
        self.label = label
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        self.features_mean = features_mean
        self.features_deviation = features_deviation

        self.unique_label = np.unique(label)  # [0 1 2]
        self.num_features = self.data.shape[1]  # 特征个数==data列数
        self.num_unique_label = self.unique_label.shape[0]  # 标签个数

        self.theta = np.zeros((self.num_unique_label, self.num_features))  # 3行5列，可能要用到转置
        # [[0. 0. 0. 0. 0.]
        #  [0. 0. 0. 0. 0.]
        #  [0. 0. 0. 0. 0.]]

    def train(self, max_iterations=1000):
        cost_histories = []  # 记录损失函数的变化历史
        for index, label in enumerate(self.unique_label):  # 针对多分类标签[0,1,2]，二分类化
            current_initial_theta = np.copy(self.theta[index].reshape(self.num_features, 1))  # 拿到当前标签的参数
            current_label = (self.label == self.unique_label).astype(float)
            print(current_label)


if __name__ == '__main__':
    data_ = load_iris().data
    label_ = load_iris().target

    lr = LogisticRegression(data_, label_)
    lr.train()
