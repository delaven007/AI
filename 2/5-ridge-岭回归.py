import sklearn.metrics as sm
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
import numpy as np

#读取文件
x,y=np.loadtxt('./data/ml_data/abnormal.txt',delimiter=',',usecols=(0,1),unpack=True)
print(x.shape,x.dtype,y.shape,y.dtype)



model=lm.LinearRegression()
x=x.reshape(-1,1)               #输入集转为n行1列
model.fit(x,y)
linex=np.linspace(x.min(),x.max(),500)
liney=model.predict(linex.reshape(-1,1))
# mp.plot(linex,liney,color='red',label='Regression Line')
#预测
pred_y=model.predict(x)


#训练岭回归模型
model=lm.Ridge(150,fit_intercept=True,max_iter=10000)
model.fit(x,y)
liney=model.predict(linex.reshape(-1,1))
mp.plot(linex,liney,color='green',label='Ridge Line')


# mp.figure('LinearRegression',facecolor='lightgray')
# mp.grid(linestyle=':')
mp.scatter(x.ravel(),y.ravel(),s=70,color='blue',label='Samples')               #画出散点
# #
mp.plot(x,pred_y,color='orange',label='Ression Line')
mp.legend()
mp.show()







