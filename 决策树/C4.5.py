"""
    ID3、C4.5决策树
"""
import math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


class Node:
    # 这里构建树的节点类，也可用字典来表示树结构
    def __init__(self, fea=-1, res=None, child=None):
        self.fea = fea
        self.res = res
        self.child = child  # 特征的每个值对应一颗子树，特征值为键，相应子树为值


class DecisionTree:
    def __init__(self, epsilon=1e-3, metric='C4.5'):
        self.tree = None
        self.epsilon = epsilon
        self.metric = metric

    def fit(self, data, label):
        fea_list = list(range(data.shape[1]))
        self.tree = self.build_tree(fea_list, data, label)
        return self.tree

    def predict(self, data):
        data = pd.DataFrame(data)
        result = data.apply(lambda x: self.predict_inner(x), axis=1)
        return result

    def predict_inner(self, data):
        def helper(data_, tree):
            if tree.res is not None:  # 表明到达叶子节点
                return tree.res
            else:
                try:
                    sub_tree = tree.child[data_[tree.fea]]
                    return helper(data_, sub_tree)
                except Exception as e:
                    print("输入数据超出维度")

        return helper(data, self.tree)

    @staticmethod
    def calculate_empirical_entropy(label):  # 计算整体经验熵
        c = pd.DataFrame(label).value_counts().tolist()
        n = len(label)
        ent = 0
        for val in c:
            p = val / n
            ent += -p * math.log2(p)
        return ent

    def calculate_condition_entropy(self, fea, data, label):
        # 根据特征fea拆分子集
        sub_data = defaultdict(list)  # 初始化一个内容为列表的字典
        for index, sample in enumerate(data):
            sub_data[sample[fea]].append(index)  # 将fea列中数值相同的行index汇集在一起

        # 计算条件熵
        c = Counter(data[:, fea])
        ent = 0
        n = len(label)
        for key, val in c.items():
            pi = val / n
            # label[sub_data[key]]：key对应的子集集合的index序号，label[序号]可以找到子集对应的标签
            # 注意：此处与上面多对应的都是字典，方便用key对应关联
            ent = ent + pi * self.calculate_empirical_entropy(label[sub_data[key]])  # 计算每一个子集的经验熵并求和
        return ent, sub_data

    def info_gain(self, fea, data, label):
        # 计算信息增益,用于ID3算法
        exp_ent = self.calculate_empirical_entropy(label)
        con_ent, sub_data = self.calculate_condition_entropy(fea, data, label)
        return exp_ent - con_ent, sub_data

    def info_gain_radio(self, fea, data, label):
        # 计算信息增益比，用于C4.5
        info_add, sub_data = self.info_gain(fea, data, label)
        n = len(label)
        split_radio = 0
        for val in sub_data.values():
            p = len(val) / n  # 每一个子集数据量占总体数据的占比
            # 以下是划分子集的占比所计算的熵，而不是整体熵的label标签的占比
            split_radio -= p * math.log2(p)
        return info_add / split_radio, sub_data

    def best_feature(self, fea_list, data, label):
        # 获取最优切分特征，相应的信息增益比以及切分后的子数据集
        score_func = self.info_gain_radio
        if self.metric == "ID3":
            score_func = self.info_gain

        best_fea = fea_list[0]  # 选择首个，将剩下的分别再计算信息增益，与首个对比，计算出最优切分特征
        g_max, best_sub_data = score_func(best_fea, data, label)
        for fea in fea_list[1:]:
            g, sub_data = score_func(fea, data, label)
            if g > g_max:
                best_fea = fea
                best_sub_data = sub_data
                g_max = g
        return g_max, best_fea, best_sub_data

    def build_tree(self, fea_list, data, label):  # 重点
        label_unique = np.unique(label)
        if label_unique.shape[0] == 1:  # 只剩下一个标签了 那就是你了
            return Node(res=label_unique[0])  # label_unique[0]=0|1|2
        if not fea_list:
            return Node(res=Counter(label).most_common(1)[0][0])
        g_max, best_fea, best_sub_data = self.best_feature(fea_list, data, label)  # 找出最好的特征以及子集
        if g_max < self.epsilon:  # 信息增益小于阈值，返回数据集中的占大部分的类
            return Node(res=Counter(label).most_common(1)[0][0])

        # TODO  此处else指的是尚处于分支，上面三个return才是最终返回的结果，下面的分支也是为了输出上面的结果
        else:
            fea_list.remove(best_fea)
            child = {}
            for key, val in best_sub_data.items():
                child[key] = self.build_tree(fea_list, data[val], label[val])
            return Node(fea=best_fea, child=child)


if __name__ == '__main__':
    x_ = load_iris().data
    y_ = load_iris().target

    clf = DecisionTree()
    clf.fit(x_, y_)
    print(clf.predict(x_))
