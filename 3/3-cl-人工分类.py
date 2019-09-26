
import numpy as np
import matplotlib.pyplot as mp

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

#绘制分类边界线（把整个的图标区间表格化，预测每个小格子的类别标签）
#使用pcolormesh为这些小格子添加标签
l,r=x[:,0].min()-1,x[:,0].max()+1
b,t=x[:,1].min()-1,x[:,1].max()+1

n=1000
grid_x,grid_y=np.meshgrid(
    np.linspace(1,r,n),
    np.linspace(b,t,n)
    )
#数组处理函数
grid_z=np.piecewise(grid_x,[grid_x>grid_y,grid_x<=grid_y],[0,1])
print(grid_z)


#画图
mp.figure('Simple Classfication',facecolor='lightgray')
mp.title('Simple Classfication',fontsize=17)
mp.grid(linestyle=':')
mp.pcolormesh(grid_x,grid_y,grid_z,cmap='gray')
mp.scatter(x[:,0],x[:,1],s=70,marker='o',c=y,cmap='jet',label='sample')
mp.legend()
mp.show()

























