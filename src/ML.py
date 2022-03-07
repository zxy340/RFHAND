import numpy as np
from data import data_loader
import os
import matplotlib.pyplot as plt

# .............load RGB and mmWave data plus the peaks information ................
if os.path.exists('../data/x_train.npy') & os.path.exists('../data/y_train.npy') \
     & os.path.exists('../data/x_test.npy') & os.path.exists('../data/y_test.npy'):
    x_train = np.load('../data/x_train.npy')
    y_train = np.load('../data/y_train.npy')
    x_test = np.load('../data/x_test.npy')
    y_test = np.load('../data/y_test.npy')
else:
    current_data_index = 11  # indicate the current loaded action index
    num_per_act = 8  # indicate the data folder number of one action
    x_train, y_train, x_test, y_test = data_loader(current_data_index, num_per_act)
    np.save('../data/x_train.npy', x_train)
    np.save('../data/y_train.npy', y_train)
    np.save('../data/x_test.npy', x_test)
    np.save('../data/y_test.npy', y_test)

y_train[:, 0] = y_train[:, 0] / 1920
y_test[:, 0] = y_test[:, 0] / 1920
y_train[:, 1] = y_train[:, 1] / 1080
y_test[:, 1] = y_test[:, 1] / 1080
x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)
# ..................................................................................

# 训练随机森林解决回归问题
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
x1 = y_pred[:, 0]
y1 = y_pred[:, 1]
x2 = y_test[:, 0]
y2 = y_test[:, 1]
plt.plot(x1, y1, 'g^', x2, y2, 'bs')
plt.show()
plt.pause(5)

# 评估回归性能
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

