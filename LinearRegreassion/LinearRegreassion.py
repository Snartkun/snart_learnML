from __future__ import print_function
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_data(filename, split, dataType):
    return np.loadtxt(filename, delimiter=split, dtype=dataType)

def linearRegression():
    print(U"加载数据...\n")
    data = load_data("data.txt", ",", np.float64)
    X = np.array(data[:,0:-1], dtype=np.float64)
    Y = np.array(data[:,-1], dtype=np.float64)

    #归一化
    scaler = StandardScaler()
    scaler.fit(X)
    x_train = scaler.transform(X)
    x_test = scaler.transform(np.array([[1940, 4]]))

    #线性模型拟合
    model = linear_model.LinearRegression()
    model.fit(x_train, Y)

    #预测结果
    result = model.predict(x_test)
    print(model.coef_)
    print(model.intercept_)
    print(result)

if __name__ == '__main__':
    linearRegression()