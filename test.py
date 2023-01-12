import math

import numpy as np

if __name__ == '__main__':
    D = 4
    L = 4
    n_fea = int(math.log2(D) + 1)  # 默认选择特征的个数
    print(n_fea)
    fea_arr = np.zeros((n_fea, D))
    for i in range(n_fea):
        out_fea = np.random.choice(D, L, replace=False)
        print(out_fea)
        coeff = np.random.uniform(-1, 1, D)  # 在[-1,1]上的均匀分布来产生每个特征前的系数
        coeff[out_fea] = 0
        fea_arr[i] = coeff

    print(fea_arr)
