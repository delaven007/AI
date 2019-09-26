import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data=np.loadtxt('./data/ml_data/multiple2.txt',delimiter=',',dtype='f8')
x=data[:,:-1]
y=data[:,-1]
print(x.shape,y.shape)

#拆分测试集与训练集
train_x,test_x,train_y,test_y=ms.train_test_split(x,y,test_size=0.25,random_state=7)
# print(train_x,test_x,train_y,test_y)
#训练svm训练器
model=svm.SVC(kernel='rbf',C=600,gamma=0.01)
model.fit(train_x,train_y)
#预测
pre_test_y=model.predict(test_x)
#输出分类结果
cr=sm.classification_report(test_y,pre_test_y)
print(cr)


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

#绘制样本空间
mp.figure('SVM', facecolor='lightgray')
mp.title('SVM Classification', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg', s=80)
mp.legend()
mp.show()














