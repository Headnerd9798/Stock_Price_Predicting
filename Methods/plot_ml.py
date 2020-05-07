import numpy as np
import pandas as pd
import matplotlib.pylab as plt
# Read result
y_true_all = pd.read_csv('./dataset/dataset_y/data_y.csv', header=0)
data_lr = pd.read_csv('./dataset/dataset_result/result_linear_regression.csv', header=0)
data_svm = pd.read_csv('./dataset/dataset_result/result_support_vector_machine.csv', header=0)
data_rf = pd.read_csv('./dataset/dataset_result/result_random_forest.csv', header=0)
data_et = pd.read_csv('./dataset/dataset_result/result_extra_tree.csv', header=0)
data_gb = pd.read_csv('./dataset/dataset_result/result_gradient_boosting.csv', header=0)
lr = np.array(data_lr)
svm = np.array(data_svm)
rf = np.array(data_rf)
et = np.array(data_et)
gb = np.array(data_gb)


# select last month data （last 161 rows）
y_lr = lr[-161:, -1]
y_svm = svm[-161:, -1]
y_rf = rf[-161:, -1]
y_et = et[-161:, -1]
y_gb = gb[-161:, -1]
x = rf[-161:, 0]
y_true_1month = y_true_all[-161:]

# plot

# Linear Regression
plt.title('Linear Regression')
plt.plot(x, y_lr, color='crimson', linewidth=1, linestyle='-',label = 'Predicted Stock Price')
plt.plot(x, y_true_1month, color='navy', linewidth=1, linestyle='-', label='True Stock Price')
plt.xticks(x[::10], rotation=45)
# label every 10
plt.xlabel('Time')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.savefig('../Figure/Linear_Regression_Month.png', bbox_inches = 'tight')
plt.show()

# Support Vector Machine
plt.title('Support Vector Machine')
plt.plot(x, y_lr, color='crimson', linewidth=1, linestyle='-',label = 'Predicted Stock Price')
plt.plot(x, y_true_1month, color='navy', linewidth=1, linestyle='-', label='True Stock Price')
plt.xticks(x[::10], rotation=45)
# label every 10
plt.xlabel('Time')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.savefig('../Figure/Support_Vector_Machine_Month.png', bbox_inches = 'tight')
plt.show()

# Random Forest
plt.title('Random Forest Regressor')
plt.plot(x, y_rf, color='crimson', linewidth=1, linestyle='-',label = 'Predicted Stock Price')
plt.plot(x, y_true_1month, color='navy', linewidth=1, linestyle='-', label = 'True Stock Price')
plt.xticks(x[::10], rotation=45)
# label every 10
plt.xlabel('Time')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.savefig('../Figure/Random_Forest_Month.png', bbox_inches = 'tight')
plt.show()

# Extra Tree
plt.title('Extra Tree Regressor')
plt.plot(x, y_et, color='crimson', linewidth=1, linestyle='-',label = 'Predicted Stock Price')
plt.plot(x, y_true_1month, color='navy', linewidth=1, linestyle='-', label = 'True Stock Price')
plt.xticks(x[::10], rotation=45)
# label every 10
plt.xlabel('Time')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.savefig('../Figure/Extra_Tree_Month.png', bbox_inches = 'tight')
plt.show()

# Gradient Boosting
plt.title('Gradient Boosting Regressor')
plt.plot(x, y_gb, color='crimson', linewidth=1, linestyle='-',label = 'Predicted Stock Price')
plt.plot(x, y_true_1month, color='navy', linewidth=1, linestyle='-', label = 'True Stock Price')
plt.xticks(x[::10], rotation=45)
# label every 10
plt.xlabel('Time')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.savefig('../Figure/Gradient_Boosting_Month.png', bbox_inches = 'tight')
plt.show()


