"""
    from：https://zhuanlan.zhihu.com/p/98061179
"""
import numpy as np
import pandas as pd


class SoftmaxRegression:
    def __init__(self, iters=1000, alpha=0.1, lam=0.01):
        self.theta = None
        self.n_classes = None
        self.n_samples = None
        self.n_features = None
        self.iteration = iters
        self.alpha = alpha
        self.lam = lam

    def fit(self, x, y):
        self.n_classes = np.unique(y).shape[0]  # 获取样本中的类别数
        self.n_samples = x.shape[0]  # 获取样本的样本量
        self.n_features = x.shape[1]  # 获取样本的特征列数

        # 初始化权重参数矩阵,这里：data*self.theta.T==>会输出多列概率，即当前x对应的不同类别的概率，是多个并非一个
        self.theta = np.random.rand(self.n_classes, self.n_features)  #

        # 创建损失的列表
        all_loss = []

        # 对标签进行one-hot编码,[0. 0. 0. 1.]，即每个标签在对应标签处为1，其余为0
        y_one_hot = self.one_hot(y, self.n_samples, self.n_classes)

        # 迭代交叉熵求参数矩阵self.theta
        for i in range(self.iteration):
            # 计算预测结果，prediction
            result = np.dot(x, self.theta.T)  # 注意这里的参数矩阵，必须转置才能进行矩阵相乘
            prediction = self.softmax(result)  # softmax函数将结果归一化，以此概率化
            # print(result[i].sum())  # 14.720170035537432,...,17.410918012636426
            # print(prediction[i].sum())  # 1.0,...,1.0  这就是softmax的作用

            # 计算损失函数，此处用的交叉熵的解法，而不是极大似然估计法,最小化损失函数
            """
            log函数：prediction<1,log<0,prediction越大，log的绝对值越小，又因为是负号，所以loss最前面加一个符号
            y_one_hot: 意味着，当y属于当前类别的时候，y_one_hot=1 else 0
            """
            loss = -(1.0 / self.n_samples) * np.sum(y_one_hot * np.log(prediction))
            all_loss.append(loss)

            # 最小化损失函数，求解梯度，这里在数据上需要推理,这里加上了正则项：+ self.lam * self.theta
            # #加入正则项之后的梯度
            # TODO : The most important step !!!
            dw = -(1.0 / self.n_samples) * np.dot((y_one_hot - prediction).T, x) + self.lam * self.theta
            dw[:, 0] = dw[:, 0] - self.lam * self.theta[:, 0]

            # 更新权重参数
            self.theta = self.theta - self.alpha * dw
        return self.theta, all_loss

    def predict(self, data):
        prediction = self.softmax(np.dot(data, self.theta.T))
        return np.argmax(prediction, axis=1).reshape((-1, 1))

    @staticmethod
    def one_hot(y, n_samples, n_classes):
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), y.T] = 1
        return one_hot

    @staticmethod
    def softmax(result):
        # 计算总和,求和归一，以此求出概率，！！关键点在于：不同类别求和得到1
        sum_exp = np.sum(np.exp(result), axis=1, keepdims=True)
        softmax = np.exp(result) / sum_exp
        return softmax


if __name__ == '__main__':
    data_ = pd.read_csv("../data/train_data.csv")
    x_ = np.array(data_.iloc[:, :-1])
    y_ = np.array(data_.iloc[:, -1]).reshape(-1, 1)
    sg = SoftmaxRegression()
    sg.fit(x_, y_)
    y_prediction = sg.predict(x_)
    accuracy = np.sum(y_prediction == y_) / len(y_)
    print(accuracy)
