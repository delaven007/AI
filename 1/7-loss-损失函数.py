
import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d

x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

w0,w1=np.meshgrid(np.linspace(-10,20,1000),np.linspace(-10,10,1000))

#根据w0与w1计算每个坐标点的loss值
loss=np.zeros(w0.shape)
for px,py in zip(x,y):
    loss+=1/2*(w0+w1*px-py)**2

mp.figure("3D Contour",facecolor='orange')
axes3d=mp.gca(projection='3d')
# axes3d.plot_surface(x,y,z,cstride=10,rstride=10,cmap='jet')
axes3d.plot_wireframe(w0,w1,loss)
mp.tight_layout()
mp.show()




















