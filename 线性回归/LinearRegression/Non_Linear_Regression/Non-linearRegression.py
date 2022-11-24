import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from 线性回归.LinearRegression.Linear_Regression.linear_regression import LinearRegression

if __name__ == '__main__':
    data = pd.read_csv('../../../线性回归/data/non-linear-regression-x-y.csv')

    x = data['x'].values.reshape((data.shape[0], 1))
    y = data['y'].values.reshape((data.shape[0], 1))

    num_iterations = 50000
    learning_rate = 0.02
    polynomial_degree = 15
    sinusoid_degree = 15
    normalize_data = True

    linear_regression = LinearRegression(x, y, polynomial_degree, sinusoid_degree, normalize_data)
    linear_regression.train(learning_rate, num_iterations)
    y_predictions = linear_regression.predict(x)

    plt.plot(x, y_predictions)
    plt.show()
