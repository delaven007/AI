
import numpy as np
import sklearn.preprocessing as sp


#标签处理一列数据
samples=np.array(['apple','banana','orange','pair','edge','elephone','fox'])
print(samples)
# 获取标签编码器
lbe=sp.LabelEncoder()
# 调用标签编码器的fit_transform方法训练并且为原始样本矩阵进行标签编码
			  #先训练在转换
lb=lbe.fit_transform(samples)
print(lb)
print(lbe.transform(['orange']))

#预测结果，把编码转成对应字符串
label=[0,1,5,6,2,3,4]
# 根据标签编码的结果矩阵反查字典 得到原始数据矩阵
w=lbe.inverse_transform(label)
print(w)



















