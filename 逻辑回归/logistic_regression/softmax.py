import numpy as np
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("../data/train_data.csv")
    x = np.array(data.iloc[:, :-1])
    y = np.array(data.iloc[:, -1]).reshape(-1, 1)

    print(x)
    print(y)
