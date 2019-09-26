import sklearn.metrics as sm
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
import numpy as np

#读取文件
x,y=np.loadtxt('./data/ml_data/single.txt',delimiter=',',usecols=(0,1),unpack=True)
print(x.shape,x.dtype,y.shape,y.dtype)

#训练模型

model=lm.LinearRegression()
x=x.reshape(-1,1)               #输入集转为n行1列
model.fit(x,y)

#预测
pred_y=model.predict(x)

# 平均绝对值误差：1/m∑|实际输出-预测输出|
print(sm.mean_absolute_error(y, pred_y))
# 平均平方误差：SQRT(1/mΣ(实际输出-预测输 出)^2)
print(sm.mean_squared_error(y, pred_y))
# 中位绝对值误差：MEDIAN(|实际输出-预测输出|)
print(sm.median_absolute_error(y, pred_y))
# R2得分，(0,1]区间的分值。分数越高，误差越小。
print(sm.r2_score(y, pred_y))




mp.figure('LinearRegression',facecolor='lightgray')
mp.grid(linestyle=':')
mp.scatter(x.ravel(),y.ravel(),s=70,color='blue',label='Samples')               #画出散点

mp.plot(x,pred_y,color='orange',label='Ression Line')
mp.legend()
mp.show()







