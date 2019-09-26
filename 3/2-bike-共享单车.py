
import numpy as np
import  sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp


data=[]
header=None
with open('./data/ml_data/bike_day.csv','r')as f:
    for i ,line in enumerate(f.readlines()):
        if i==0:
            header=line[::-1].split(',')
        else:
            data.append(line.split(','))

#整理数据集
header=header[2:13]
# print(header)
data=np.array(data)
x=data[:,2:13].astype('f8')
y=data[:,-1].astype('f8').ravel()            #将二维数组拉成一维数组


#打乱数据集，拆分训练集与测试集
x,y=su.shuffle(x,y,random_state=7)
train_size=int(len(x)*0.9)
train_x,test_x,train_y,test_y=x[:train_size],x[train_size:],y[:train_size],y[train_size:]
#训练随机森林回归模型，预测结果
model=se.RandomForestRegressor(max_depth=10,n_estimators=1000,min_samples_split=2)
model.fit(train_x,train_y)
predict_test_y=model.predict(test_x)
#评估模型
print(sm.r2_score(test_y,predict_test_y))










