import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Data\数据分析.csv")
    train_x = data.iloc[:190, 0]
    train_y = data.iloc[:190, 1]
    test_x = data.iloc[190:, 0]
    test_y = data.iloc[190:, 1]

    lr = LinearRegression()
    lr.fit(np.array(train_x).reshape(-1, 1), train_y)
    result = np.array(lr.predict(np.array(test_x).reshape(-1, 1))).reshape(-1, 1)
    test_y = np.array(test_y).reshape(-1, 1)
    result = np.hstack((test_y, result))

    result = pd.DataFrame(data=result, columns=["label", "prediction"])

    mse = mean_squared_error(y_true=result["label"], y_pred=result["prediction"])
    print(mse)
