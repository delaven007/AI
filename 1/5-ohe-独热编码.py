

import sklearn.preprocessing as sp
import numpy as np
import matplotlib.pyplot as mp
import scipy.misc as sm

samples=np.array([[1,3,2],
                  [7,5,4],
                  [1,8,6],
                  [7,3,6]]
                 )

ohe=sp.OneHotEncoder(sparse=False)
r=ohe.fit_transform(samples)
print(r,type(r))

















