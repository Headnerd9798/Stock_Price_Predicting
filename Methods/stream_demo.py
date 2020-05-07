import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation


y_true_all = pd.read_csv('../dataset/dataset_y/data_y.csv', header=0)
data_rf = pd.read_csv('../dataset/dataset_result/result_random_forest.csv', header=0)
rf = np.array(data_rf)
y_rf = np.array(rf[-161:, -1])
y_true_1month = np.array(y_true_all[-161:])
x = np.array(rf[-161:, 0])


fig, ax = plt.subplots()
t, p_t, p_p = [], [], []
line, = ax.plot([], [], color='navy', linewidth=1, linestyle='-', label='True Stock Price', animated=False)
line2, = ax.plot([], [], color='crimson', linewidth=1, linestyle='-', label='Predicted Stock Price', animated=False)


def init():
    ax.set_xlim(0, 160)
    ax.set_ylim(100, 180)
    return line,


def update(frame):
    t.append(frame)
    p_t.append(y_true_1month[frame])
    p_p.append(y_rf[frame])
    line.set_data(t, p_t)
    line2.set_data(t, p_p)
    return line, line2

ani = FuncAnimation(fig, update, frames=160,init_func=init, blit=True, repeat= False)
plt.title('Random Forest Regressor')
plt.xlabel('Time(Hour)')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.show()

