
import sklearn.preprocessing as sp
import numpy as np


samples=np.array([[17.,90.,6000.],
                  [20.,100.,8000.],
                  [25.,70.,7000.]])

mms=sp.MinMaxScaler(feature_range=(0,1))
r=mms.fit_transform(samples)
print(r)

#手动实现
samples_copy=[]
for col in samples.T:
    col_min=col.min()
    col_max=col.max()
    #[[max 1][max 1] * [k, b]=[0 1]]
    A=np.array([[col_min,1],[col_max,1]])
    B=np.array([0,1])
    #解方程        ==lstsq
    x=np.linalg.solve(A,B)
    k,b=x[0],x[1]
    y=col * k + b
    samples_copy.append(y)
print(np.array(samples_copy).T)






















