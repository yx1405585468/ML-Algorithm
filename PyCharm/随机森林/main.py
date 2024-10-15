import math
from collections import defaultdict

from sklearn.datasets import load_iris
import numpy as np

from 决策树.cart_clf import CART_CLF


def validate(x, y_, ratio=0.2):
    n = x.shape[0]
    size = int(n * ratio)
    index_s = np.random.permutation(range(n))
    for i in range(int(n / size)):
        test_index = index_s[i * size:(i + 1) * size]
        train_index = list(set(range(n)) - set(test_index))
        yield x[train_index], y_[train_index], x[test_index], y_[test_index]


class RandomForest:
    def __init__(self, n_tree=6, n_fea=None, ri_rc=True, L=None, epsilon=1e-3, min_sample=1):
        self.n_tree = n_tree  # 森林中有多少棵树
        self.n_fea = n_fea  # 每棵树中特征的数量
        self.ri_rc = ri_rc  # 判定特征的选择选用RI还是RC,特征比较少使用RC
        self.L = L  # 选择RC时，进行线性组合的特征的个数
        self.tree_list = []  # 随机森林中子树的list

        self.epsilon = epsilon
        self.min_sample = min_sample  # 叶子节点含有的最少样本数

        self.D = None  # 输入数据的维度
        self.N = None

    def extract_fea(self):
        # 从原数据中抽取特征(RI)或线性组合构建新特征(RC)
        if self.ri_rc:  # 特征较多时，RI提取
            if self.n_fea > self.D:
                raise ValueError("使用RI提取时，选择的特征个数必须小于数据的维度数")
            fea_arr = np.random.choice(self.D, self.n_fea, replace=False)

        else:  # 特征较少时，RC提取
            fea_arr = np.zeros((self.n_fea, self.D))
            for i in range(self.n_fea):
                out_fea = np.random.choice(self.D, self.L, replace=False)
                coe_ff = np.random.uniform(-1, 1, self.D)  # 在[-1,1]上的均匀分布来产生每个特征前的系数
                coe_ff[out_fea] = 0
                fea_arr[i] = coe_ff
        return fea_arr

    def extract_data(self, x, y_):
        # 从原数据集中有放回的抽取样本，构成每个决策树的自助样本集
        fea_arr = self.extract_fea()  # col_index or coe_ffs
        index_s = np.unique(np.random.choice(self.N, self.N))  # 部分数据集的index
        subset_x = x[index_s]
        subset_y = y_[index_s]
        if self.ri_rc:
            subset_x = subset_x[:, fea_arr]
        else:
            subset_x = subset_x @ fea_arr.T
        return subset_x, subset_y, fea_arr

    def fit(self, x, y_):
        # 初始化参数
        self.D = x.shape[1]  # 特征数（维度）
        self.N = x.shape[0]  # 个数
        if self.n_fea is None:
            self.n_fea = int(math.log2(self.D) + 1)  # 默认选择特征的个数

        # 训练主函数
        for _ in range(self.n_tree):
            sub_x, sub_y, fea_arr = self.extract_data(x, y_)
            subtree = CART_CLF(epsilon=self.epsilon, min_sample=self.min_sample)
            subtree.fit(sub_x, sub_y)
            self.tree_list.append((subtree, fea_arr))  # 保存训练后的树及其选用的特征，以便后续预测时使用

    def predict(self, x):
        # 预测，多数表决
        res = defaultdict(int)  # 存储每个类得到的票数
        for i in self.tree_list:
            subtree, fea_arr = i
            if self.ri_rc:
                x_modify = x[fea_arr]
            else:
                x_modify = (np.array([x]) @ fea_arr.T)[0]
            label = subtree.predict(x_modify)
            res[label] += 1
        return max(res, key=res.get)


if __name__ == '__main__':
    data = load_iris()
    x_data = data["data"]
    y_data = data["target"]
    g = validate(x_data, y_data, ratio=0.2)
    for item in g:
        x_train, y_train, x_test, y_test = item
        RF = RandomForest(n_tree=50, n_fea=2, ri_rc=True)
        RF.fit(x_train, y_train)
        score = 0
        for X, y in zip(x_test, y_test):
            if RF.predict(X) == y:
                score += 1
        print(score / len(y_test))
