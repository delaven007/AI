
import sklearn.preprocessing as sp
import numpy as np
import matplotlib.pyplot as mp
import scipy.misc as sm

samples=np.array([[17.,90.,6000.],
                  [20.,100.,8000.],
                  [25.,70.,7000.]])

bin=sp.Binarizer(threshold=80)
#二值化器
r=bin.transform(samples)
print(r)

#读取数据
img=sm.imread('../data/da_date/lily.jpg',True)
#转置
img2=bin.transform(img)
mp.imshow(img2,cmap='gray')
mp.subplot(121)

img[img<=120]=0
img[img>120]=1
mp.subplot(122)
mp.imshow(img,cmap='gray')
mp.show()



















