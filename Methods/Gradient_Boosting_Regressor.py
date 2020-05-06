import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# Data clean
raw_data = pd.read_csv('./dataset/cleaned-data-1.csv', header=0)
clean_data = raw_data.dropna(axis=0) # drop rows that contains null value
data_df = np.array(clean_data)
# print(len(data_df))
# print(data_df[:, 2:9])
data_df_x = data_df[:, 2:10]     # extract all 8 features
data_df_y = data_df[:, 10]       # extract true stock price of ZOOM
# print(data_df_x)
# print(data_df_y)
np.savetxt('./dataset/dataset_x/data_x.csv', data_df_x.astype(float), delimiter=",")
np.savetxt('./dataset/dataset_y/data_y.csv', data_df_y.astype(float), delimiter=",")


def gradient_boosting(x_train, y_train, x_test):
    gbr = GradientBoostingRegressor()
    gbr.fit(x_train, y_train)
    y_pred_gbr = gbr.predict(x_test.reshape(1, -1))
    return y_pred_gbr


# stream data
def func_gbr(x):
    result_gbr = x[0:30, 10]
    for i in range(0, len(x) - 30):
        x_train = x[i:i+30, 2:10]
        y_train = x[i:i+30, 10]
        x_test = x[i+30, 2:10]
        y_test = x[i+30, 10]
        y_gbr = gradient_boosting(x_train, y_train, x_test)
        result_gbr = np.append(result_gbr, y_gbr)
    return result_gbr


result_gbr = func_gbr(data_df)

out = np.vstack((data_df[:, 1], result_gbr))

output = np.transpose(out)

np.savetxt('./dataset/dataset_result/result_gradient_boosting.csv', output, delimiter=",", fmt='%s')
