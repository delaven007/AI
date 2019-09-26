
import numpy as np
import matplotlib.pyplot as mp

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

# 梯度下降
times = 1000
lrate = 0.01  # 学习率
w0, w1,losses = [1], [1],[]
epoches=[]
for i in range(1, times + 1):
    epoches.append(i)
    loss = (((w0[-1] + w1[-1] * train_x) - train_y) ** 2).sum() / 2
    losses.append(loss)
    # 求取d0与d1，两方向的偏导数
    d0 = (w0[-1] + w1[-1] * train_x - train_y).sum()
    d1 = (train_x * (w0[-1] + w1[-1] * train_x - train_y)).sum()
    #输出w0,w1,loss的值
    print('{:4}> w0={:.6f}, w1={:.6f}, loss={:.6f}'.format(epoches[-1], w0[-1], w1[-1], losses[-1]))
    w0.append(w0[-1]-d0 * lrate)
    w1.append(w1[-1]-d1 * lrate)
print(w0[-1],w1[-1])


#绘图
# mp.figure('Linear Regreesion',facecolor='lightgray')
# mp.scatter(train_x,train_y,s=70,marker='o',color='blue',label='Sample Points')

#绘制回归线
# pred_y=w0[-1] + w1[-1] *train_x
# mp.plot(train_x,pred_y,color='orange',label='Regression Line')


# mp.legend()

#绘制w0函数变化曲线
# mp.figure('Training Process',facecolor='orange')
# mp.subplot(3,1,1)
# mp.grid(linestyle=':')
# mp.ylabel(r'$w_0$',fontsize=14)
# mp.plot(epoches,w0[:-1],color='blue',label=r'$w_0$')
# mp.legend()

#绘制w1函数变化曲线
# mp.figure('Training Process',facecolor='orange')
# mp.subplot(3,1,2)
# mp.grid(linestyle=':')
# mp.ylabel(r'$w_1$',fontsize=14)
# mp.plot(epoches,w1[:-1],color='gold',label=r'$w_1$')
# mp.legend()

#绘制losses函数变化曲线
# mp.figure('Training Process',facecolor='orange')
# mp.subplot(3,1,3)
# mp.grid(linestyle=':')
# mp.ylabel(r'$losses$',fontsize=14)
# mp.plot(epoches,losses,color='red',label=r'$losses$')
# mp.legend()

#绘制三维曲面（基于三维曲面绘制梯度下降过程中的每一个点。）
import mpl_toolkits.mplot3d as axes3d

grid_w0, grid_w1 = np.meshgrid(
    np.linspace(0, 9, 500),
    np.linspace(0, 3.5, 500))

grid_loss = np.zeros_like(grid_w0)
for x, y in zip(train_x, train_y):
    grid_loss += ((grid_w0 + x*grid_w1 - y) ** 2) / 2

mp.figure('Loss Function')
ax = mp.gca(projection='3d')
mp.title('Loss Function', fontsize=20)
ax.set_xlabel('w0', fontsize=14)
ax.set_ylabel('w1', fontsize=14)
ax.set_zlabel('loss', fontsize=14)
ax.plot_surface(grid_w0, grid_w1, grid_loss, rstride=10, cstride=10, cmap='jet')
ax.plot(w0[:-1], w1[:-1], losses, 'o-', c='orangered', label='BGD')                 #
mp.legend()

#以等高线的方式绘制梯度下降的过程
mp.figure('Batch Gradient Descent', facecolor='lightgray')
mp.title('Batch Gradient Descent', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.contourf(grid_w0, grid_w1, grid_loss, 10, cmap='jet')
cntr = mp.contour(grid_w0, grid_w1, grid_loss, 10,
                  colors='black', linewidths=0.5)
mp.clabel(cntr, inline_spacing=0.1, fmt='%.2f',
          fontsize=8)
mp.plot(w0, w1, 'o-', c='orangered', label='BGD')
mp.legend()




mp.tight_layout()
mp.show()


