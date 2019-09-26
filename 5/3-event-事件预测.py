import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import sklearn.preprocessing as sp

class DigitEncoder():
    def fit_transform(self,y):
        return y.astype('i4')
    def transform(self,y):
        return y.astype('i4')
    def inverse_transform(self,y):
        return y.astype('str')

#读取数据，整理样本集
data=[]
with open('../data/ml_data/event.txt','r') as f:
    for i, row in enumerate(f.readlines()):
        data.append(row.split(','))
data=np.array(data)


#整理数据集
data=np.delete(data,1,axis=1)
print(data.shape,data[0])

#遍历每一列，为每一列做编码，整理适合训练的样本
data=data.T
x,y=[],[]
encoders=[]
for row in range(len(data)):
    #确定当前数组使用那种encoder进行编码
    if data[row][0].isdigit():              #判断字符串是否是数字字符串
        encoder=DigitEncoder()
    else:
        encoder=sp.LabelEncoder()
    #整理输入输出集
    if row <len(data)-1:
        x.append(encoder.fit_transform(data[row]))
    else:
        y=encoder.fit_transform(data[row])
    encoders.append(encoder)
x=np.array(x).T
y=np.array(y).T
print(x[0],x.shape,y[0],x.shape)

#选择svm模型进行分类模型训练
model=svm.SVC(kernel='rbf',class_weight='balanced')
train_x,test_x,train_y,test_y=ms.train_test_split(x,y,test_size=0.25,random_state=7)
model.fit(train_x,train_y)
pred_test_y=model.predict(test_x)
print((test_y==pred_test_y).sum()/test_y.size)
print(sm.classification_report(test_y,pred_test_y))

#模拟真实环境
data=[['Tuesday', '13:30:00', '21', '23']]
data=np.array(data).T
test_x=[]
for row in range(len(data)):
    encoder=encoders[row]
    test_x.append(encoder.transform(data[row]))
test_x=np.array(test_x).T
pred_test_y=model.predict(test_x)
print(encoders[-1].inverse_transform(pred_test_y))




