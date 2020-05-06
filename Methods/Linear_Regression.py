import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,r2_score,confusion_matrix

# Data clean
raw_data = pd.read_csv('./dataset/cleaned-data-1.csv', header=0)
clean_data = raw_data.dropna(axis=0) # drop rows that contains null value
data_df = np.array(clean_data)


def linear_regression(x_train, y_train, x_test):
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test.reshape(1, -1))
    return y_pred_lr


# stream data
def func_lr(x):
    result_lr = x[0:30, 10]
    for i in range(0, len(x) - 30):
        x_train = x[i:i+30, 2:10]
        y_train = x[i:i+30, 10]
        x_test = x[i+30, 2:10]
        y_test = x[i+30, 10]
        y_lr = linear_regression(x_train, y_train, x_test)
        result_lr = np.append(result_lr, y_lr)
    return result_lr


result_lr = func_lr(data_df)

out = np.vstack((data_df[:, 1], result_lr))

output = np.transpose(out)

np.savetxt('./dataset/dataset_result/result_linear_regression.csv', output, delimiter=",", fmt='%s')



