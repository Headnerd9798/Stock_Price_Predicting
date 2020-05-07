import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# Read result
y_true = pd.read_csv('/Users/haoran_zhang/PycharmProjects/ELEN6889/Project/dataset/dataset_result/data_true.csv', header=0)
y_lstm = pd.read_csv('/Users/haoran_zhang/PycharmProjects/ELEN6889/Project/dataset/dataset_result/data_lstm.csv', header=0)
data_rf = pd.read_csv('/Users/haoran_zhang/PycharmProjects/ELEN6889/Project/dataset/dataset_result/result_random_forest.csv', header=0)
rf = np.array(data_rf)
x = rf[-33:, 0]


plt.title('Long Short-Term Memory')
plt.plot(x, y_lstm, color='crimson', linewidth=1, linestyle='-',label = 'Predicted Stock Price')
plt.plot(x, y_true, color='navy', linewidth=1, linestyle='-', label='True Stock Price')
plt.xticks(x[::5], rotation=45)
# label every 10
plt.xlabel('Time')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.savefig('../Figure/Linear_Regression_Month.png', bbox_inches = 'tight')
plt.show()