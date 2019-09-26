import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
# 采集数据
x, y = np.loadtxt('./data/ml_data/abnormal.txt', delimiter=',', usecols=(0,1), unpack=True)
x = x.reshape(-1, 1)
# 创建线性回归模型
model = lm.LinearRegression()
# 训练模型
model.fit(x, y)
# 根据输入预测输出
pred_y1 = model.predict(x)
# 创建岭回归模型
model = lm.Ridge(150, fit_intercept=True, max_iter=10000)
# 训练模型
model.fit(x, y)
# 根据输入预测输出
pred_y2 = model.predict(x)

mp.figure('Linear & Ridge', facecolor='lightgray')
mp.title('Linear & Ridge', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x, y, c='dodgerblue', alpha=0.75,
           s=60, label='Sample')
sorted_indices = x.T[0].argsort()
mp.plot(x[sorted_indices], pred_y1[sorted_indices],
        c='orangered', label='Linear')
mp.plot(x[sorted_indices], pred_y2[sorted_indices],
        c='limegreen', label='Ridge')
mp.legend()
mp.show()