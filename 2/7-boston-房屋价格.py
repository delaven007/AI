
import sklearn.datasets as sd
import sklearn.utils as su
import  sklearn.tree as st
import  sklearn.metrics as sm


#读取数据
boston=sd.load_boston()
# print(boston.feature_names)
# print(boston.data[0],boston.data.shape)
# print(boston.target[0],boston.target.shape)

#打乱数据集后，拆分训练集与测试集
x,y=su.shuffle(boston.data,boston.target,random_state=7)
print(x,y)
train_size=int(len(x)*0.8)
print(train_size)
train_x,test_x,train_y,test_y=x[:train_size],x[train_size:],y[:train_size],y[train_size:]
print(train_x,test_x,train_y,test_y)
#构建决策树回归器模型，训练模型
model=st.DecisionTreeRegressor(max_depth=5)
# print(model)
model.fit(train_x,train_y)
# print(model)
pred_test_y=model.predict(test_x)
# print(pred_test_y)
pred_train_y=model.predict(train_x)
# print(pred_train_y)
#输出r2得分
print(sm.r2_score(test_y,pred_test_y))
print(sm.r2_score(train_y,train_y))

























