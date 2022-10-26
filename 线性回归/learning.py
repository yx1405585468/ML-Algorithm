import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

if __name__ == '__main__':
    # TODO: 线性方程 y=β0+β1*x

    # 1. 在0-10之间生成200个x点
    n_sample = 200
    x = np.linspace(0, 10, n_sample)
    # print(x)  # <class 'numpy.ndarray'>

    # 2. 为截距β0 增加参数1：y=β0*1+β1*x
    X = sm.add_constant(x)
    # print(X)

    # 3. 参数β0,β1分别设置成2,5
    beta = np.array([2, 5])
    # print(beta)

    # 4. 添加误差e
    e = np.random.normal(size=n_sample)
    # print(e)

    # 5. 计算出真实值
    y_true = np.dot(X, beta) + e
    # print(y_true)

    # 6. 使用普通最小二乘法，根据x,y拟合出线性方程
    model = sm.OLS(y_true, x)  # 普通最小二乘法OLS
    res = model.fit()

    # 7. 打印拟合模型的系数β0，β1
    # print(res.params)  # [2.03751066 4.98431357]

    # 8. 打印出全部结果
    # print(res.summary())

    # 9. 输出预测值y_pre
    y_pre = res.fittedvalues
    # print(y_pre)

    # 10. 画出预测值与真实值的可视化对比图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y_true, 'o', label='data')  # 原始数据
    ax.plot(x, y_pre, 'r--.', label='test')  # 拟合数据
    ax.legend(loc='best')
    # plt.show()

    # 11. 高阶回归
    # Y=5+2⋅X+3⋅X^2
    n_sample = 200
    x = np.linspace(0, 10, n_sample)
    X = np.column_stack((x, x ** 2))
    X = sm.add_constant(X)
    beta = np.array([5, 2, 3])
    e = np.random.normal(size=n_sample)
    y = np.dot(X, beta) + e
    model = sm.OLS(y, X)
    results = model.fit()
    print(results.params)

    # 12. 高阶回归可视化
    y_fitted = results.fittedvalues
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, 'o', label='data')
    ax.plot(x, y_fitted, 'r--.', label='OLS')
    ax.legend(loc='best')
    plt.show()
