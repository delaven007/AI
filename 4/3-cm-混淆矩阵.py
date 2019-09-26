
import numpy as np
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp

import sklearn.model_selection as ms
import sklearn.metrics as sm


#不拆包
data = np.loadtxt('./data/ml_data/multiple1.txt', unpack=False, dtype='U20', delimiter=',')
print(data.shape)
#所有行的前两列
x = np.array(data[:, :-1], dtype=float)
print(x.shape)
#所有行的最后一列
y = np.array(data[:, -1], dtype=float)
print(y.shape)

#拆分训练集与测试集
train_x,test_x,train_y,test_y=ms.train_test_split(x,y,test_size=0.25,random_state=7)



# 创建高斯分布朴素贝叶斯分类器
model = nb.GaussianNB()
sc=ms.cross_val_score(model,train_x,train_y,cv=5,scoring='accuracy')
print(sc.mean())
#训练
model.fit(train_x, train_y)

#计算模型预测准确度
pred_test_y=model.predict(test_x)

#输出混淆矩阵
cm=sm.confusion_matrix(test_y,pred_test_y)
print(cm)

mp.figure('Confusion Matrix',facecolor='lightgray')
mp.xticks([])
mp.yticks([])
mp.imshow(cm,cmap='gray')




#拆分
l, r = x[:, 0].min() - 1, x[:, 0].max() + 1
b, t = x[:, 1].min() - 1, x[:, 1].max() + 1
#把总体切成500份
n = 500

#绘制分类边界线
grid_x, grid_y = np.meshgrid(np.linspace(l, r, n), np.linspace(b, t, n))
samples = np.column_stack((grid_x.ravel(), grid_y.ravel()))
grid_z = model.predict(samples)
#变维(改变成二位矩阵)
grid_z = grid_z.reshape(grid_x.shape)

# mp.figure('Naive Bayes Classification', facecolor='lightgray')
# mp.title('Naive Bayes Classification', fontsize=20)
# mp.xlabel('x', fontsize=14)
# mp.ylabel('y', fontsize=14)
# mp.tick_params(labelsize=10)
# mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
# mp.scatter(test_x[:, 0], test_x[:, 1], c=test_y, cmap='brg', s=80)
mp.show()
