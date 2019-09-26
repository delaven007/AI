
import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms

data=np.loadtxt('./data/ml_data/car.txt',delimiter=',',unpack=False,dtype='U10')
print(data.shape)

data=data.T
train_x,train_y=[],[]
encoders=[]
for row in range(len(data)):
    encoder=sp.LabelEncoder()
    if row <len(data)-1:        #读取到一列输入数据
        train_x.append(encoder.fit_transform(data[row]))
    else:
        train_y=encoder.fit_transform(data[row])
    encoders.append(encoder)


train_x=np.array(train_x).T
train_y=np.array(train_y)
print(train_x.shape,train_y.shape)

#准备随机森林分类器 ，训练分类模型
model=se.RandomForestClassifier(max_depth=6,n_estimators=200,random_state=7)

#交叉验证看模型效果
cv_score=ms.cross_val_score(model,train_x,train_y,cv=5,scoring='f1_weighted')
print(cv_score.mean())


#训练模型
model.fit(train_x,train_y)

data = [
    ['high', 'med', '5more', '4', 'big', 'low', 'unacc'],
    ['high', 'high', '4', '4', 'med', 'med', 'acc'],
    ['low', 'low', '2', '4', 'small', 'high', 'good'],
    ['low', 'med', '3', '4', 'med', 'high', 'vgood']]

#整理测试数据的输入集与输出集
data=np.array(data).T
test_x,test_y=[],[]
for row in range(len(data)):
    encoder=encoders[row]
    if row < len(data)-1:

        #从encoders列表中那拿到当时训练好的编码器
        test_x.append(encoder.transform(data[row]))
    else:
        test_y=encoder.transform(data[row])

test_x=np.array(test_x).T               #转置（行变列）
test_y=np.array(test_y)
#针对测试集，使用训练好的模型预测输出
pred_test_y=model.predict(test_x)
print(encoders[-1].inverse_transform(test_y))
print(encoders[-1].inverse_transform(pred_test_y))









