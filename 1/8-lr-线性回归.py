import numpy as np
import matplotlib.pyplot as mp

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

# 梯度下降
times = 1000000
lrate = 0.01  # 学习率
w0, w1 = 1, 1

for i in range(1, times + 1):
    # 求取d0与d1，两方向的偏导数
    d0 = (w0 + w1 * train_x - train_y).sum()
    d1 = (train_x * (w0 + w1 * train_x - train_y)).sum()
    w0=w0-d0 * lrate
    w1=w1-d1 * lrate
print(w0,w1)


#绘图
mp.figure('Linear Regreesion',facecolor='lightgray')
mp.scatter(train_x,train_y,s=70,marker='o',color='blue',label='Sample Points')

#绘制回归线
pred_y=w0 + w1 *train_x
mp.plot(train_x,pred_y,color='orange',label='Regression Line')


mp.legend()
mp.show()



