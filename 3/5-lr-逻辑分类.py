
import numpy as np
import matplotlib.pyplot as mp
import sklearn.linear_model as lm

x = np.array([
    [3, 1],
    [2, 5],
    [1, 8],
    [6, 4],
    [5, 2],
    [3, 5],
    [4, 7],
    [4, -1]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

#基于逻辑分类器，训练分类模型
model=lm.LogisticRegression(solver='liblinear',C=1)
model.fit(x,y)

#绘制分类边界线（把整个的图标区间表格化，预测每个小格子的类别标签）
#使用pcolormesh为这些小格子添加标签
l,r=x[:,0].min()-1,x[:,0].max()+1
b,t=x[:,1].min()-1,x[:,1].max()+1

n=500
grid_x,grid_y=np.meshgrid(
    np.linspace(1,r,n),
    np.linspace(b,t,n)
    )
#基于model对象，对每个坐标点进行类别预测，从而得到类别标签，使用该类别值填充背景颜色
samples=np.column_stack((grid_x.ravel(),
                         grid_y.ravel()
                         ))
grid_z =model.predict(samples)
grid_z=grid_z.reshape(grid_x.shape)
#画图
mp.figure('Simple Classfication',facecolor='lightgray')
mp.title('Simple Classfication',fontsize=17)
mp.grid(linestyle=':')
mp.pcolormesh(grid_x,grid_y,grid_z,cmap='gray')
mp.scatter(x[:,0],x[:,1],s=70,marker='o',c=y,cmap='jet',label='sample')
mp.legend()
mp.show()

























