import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,r2_score,confusion_matrix

# Data clean
raw_data = pd.read_csv('./dataset/cleaned-data-1.csv', header=0)
clean_data = raw_data.dropna(axis=0) # drop rows that contains null value
data_df = np.array(clean_data)


def support_vector_machine(x_train, y_train, x_test):
    svm = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svm.fit(x_train, y_train)
    y_pred_svm = svm.predict(x_test.reshape(1, -1))
    return y_pred_svm

# stream data
def func_svm(x):
    result_svm = x[0:30, 10]
    for i in range(0, len(x) - 30):
        x_train = x[i:i+30, 2:10]
        y_train = x[i:i+30, 10]
        x_test = x[i+30, 2:10]
        y_test = x[i+30, 10]
        y_svm = support_vector_machine(x_train, y_train, x_test)
        result_svm = np.append(result_svm, y_svm)
    return result_svm


result_svm = func_svm(data_df)

out = np.vstack((data_df[:, 1], result_svm))

output = np.transpose(out)

np.savetxt('./dataset/dataset_result/result_support_vector_machine.csv', output, delimiter=",", fmt='%s')

