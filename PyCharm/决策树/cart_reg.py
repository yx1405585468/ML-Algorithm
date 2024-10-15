"""
    CART+最小二乘法构建CART回归树
"""
import numpy as np


class Node:
    def __init__(self, fea=-1, val=None, res=None, right=None, left=None):
        self.fea = fea
        self.val = val
        self.res = res
        self.right = right
        self.left = left


class CartReg:
    def __init__(self, epsilon=0.1, min_sample=10):
        self.epsilon = epsilon
        self.min_sample = min_sample
        self.tree = None

    @staticmethod
    def error(y):
        # 子数据集的输出变量y与均值的差的平方和
        return y.var() * y.shape[0]

    @staticmethod
    def leaf(y):
        # 叶子节点取值，为了数据集输出y的均值
        return y.mean()

    @staticmethod
    def split(fea, val, x):
        # 根据某个特征，以及特征下的某个取值，将数据进行切分
        # 直接操纵index
        set1_index = np.where(x[:, fea] <= val)[0]
        set2_index = list(set(range(x.shape[0])) - set(set1_index))
        return set1_index, set2_index

    def get_best_spilt(self, x, y):
        # 求最优切分点
        best_error = self.error(y)
        best_split = None
        subset_index = None

        # 遍历所有特征，选择切分点
        for fea in range(x.shape[1]):
            for val in x[:, fea]:
                set1_index, set2_index = self.split(fea, val, x)
                # 若切分后某个子集的大小不足2，则不切分
                if len(set1_index) < 2 or len(set2_index) < 2:
                    continue
                now_error = self.error(y[set1_index]) + self.error(y[set2_index])
                if now_error < best_error:
                    best_error = now_error
                    best_split = (fea, val)
                    subset_index = (set1_index, set2_index)
        return best_error, best_split, subset_index

    def build_tree(self, x, y):
        # 构建递归二叉树
        if y.shape[0] < self.min_sample:
            return Node(res=self.leaf(y))
        best_error, best_split, subset_index = self.get_best_spilt(x, y)
        if subset_index is None:
            return Node(res=self.leaf(y))
        if best_error < self.epsilon:
            return Node(res=self.leaf(y))

        else:
            left = self.build_tree(x[subset_index[0]], y[subset_index[0]])
            right = self.build_tree(x[subset_index[1]], y[subset_index[1]])
            return Node(fea=best_split[0], val=best_split[1], right=right, left=left)

    def fit(self, x, y):
        self.tree = self.build_tree(x, y)

    def predict(self, x):
        # 对输入变量进行预测
        def helper(x, tree):
            if tree.res is not None:
                return tree.res
            else:
                if x[tree.fea] <= tree.val:
                    branch = tree.left
                else:
                    branch = tree.right
                return helper(x, branch)

        return helper(x, self.tree)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X_data_raw = np.linspace(-3, 3, 50)
    np.random.shuffle(X_data_raw)
    y_data = np.sin(X_data_raw)
    X_data = np.transpose([X_data_raw])
    y_data = y_data + 0.1 * np.random.randn(y_data.shape[0])
    clf = CartReg(epsilon=1e-4, min_sample=1)
    clf.fit(X_data, y_data)
    res = []
    for i in range(X_data.shape[0]):
        res.append(clf.predict(X_data[i]))
    p1 = plt.scatter(X_data_raw, y_data)
    p2 = plt.scatter(X_data_raw, res, marker='*')
    plt.legend([p1, p2], ['real', 'pred'], loc='upper left')
    plt.show()
