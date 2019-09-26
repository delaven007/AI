
import sklearn.preprocessing as sp
import numpy as np


samples=np.array([[17.,90.,6000.],
                  [20.,100.,8000.],
                  [25.,70.,7000.]])

r=sp.normalize(samples,norm='l2')
print(r)



