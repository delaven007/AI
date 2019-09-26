import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as mp

data=np.loadtxt('../data/ml_data/multiple2.txt',delimiter=',',dtype='f8')
x=data[:,:-1]
y=data[:,-1]
print(x.shape,y.shape)

#拆分测试集与训练集
train_x,test_x,train_y,test_y=ms.train_test_split(x,y,test_size=0.25,random_state=7)
# print(train_x,test_x,train_y,test_y)
#训练svm训练器
model=svm.SVC(kernel='rbf',C=600,gamma=0.1,probability=True)
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
mp.scatter(test_x[:, 0], test_x[:, 1], c=test_y, cmap='brg', s=80)

#新增样本点，输出置信概率
prob_x = np.array([
    [2, 1.5],
    [8, 9],
    [4.8, 5.2],
    [4, 4],
    [2.5, 7],
    [7.6, 2],
    [5.4, 5.9]])
#预测每个样本的类别
pred_prob_y=model.predict(prob_x)
#输出每个样本的置信概率
probs=model.predict_proba(prob_x)
print(probs)
# 绘制每个测试样本，并给出标注
mp.scatter(prob_x[:,0], prob_x[:,1], c=pred_prob_y, cmap='jet_r', s=80, marker='D')
for i in range(len(probs)):
    mp.annotate(
        '{}% {}%'.format(
            round(probs[i, 0] * 100, 2),
            round(probs[i, 1] * 100, 2)),
        xy=(prob_x[i, 0], prob_x[i, 1]),
        xytext=(12, -12),
        textcoords='offset points',
        horizontalalignment='left',
        verticalalignment='top',
        fontsize=9,
        bbox={'boxstyle': 'round,pad=0.6',
              'fc': 'orange', 'alpha': 0.8})

mp.scatter(prob_x[:,0],prob_x[:,1],c=pred_prob_y,s=80,marker='D',cmap='gray_r',label='prob smaples')


mp.legend()
mp.show()














