"""
    ID3、C4.5决策树算法
    ...
    决策树算法涉及到 树的数据结构
    ...
"""
import numpy as np
from sklearn.datasets import load_iris


class Node:
    # 构建树的节点类，也可以用字典来表述树的结构
    def __init__(self, fea=-1, res=None, child=None):
        self.fea = fea
        self.res = res
        self.child = child  # 特征的每个值对应一棵子树，特征值为键，相应子树为值


class DecisionTree:
    def __init__(self, method='C4.5'):
        self.tree = None
        self.method = method  # 选用ID3或者C4.5策略构造决策树

    def fit(self, data, label):
        fea_list = list(range(data.shape[1]))  # 生成特征列数列表[0, 1, 2, 3]
        self.tree = self.buildTree(fea_list, data, label)  # 根据数据集构建决策树

    def predict(self, data):
        pass

    def buildTree(self, fea_list, data, label):
        # 构建递归树
        label_unique = np.unique(label)  # 输出数据集的几种类别

        if label_unique.shape[0] == 1:  # 数据集只有一个类，直接返回该类
            return Node(res=label_unique[0])


if __name__ == '__main__':
    x_ = load_iris().data
    y_ = load_iris().target
    tree = DecisionTree()
    tree.fit(x_, y_)
