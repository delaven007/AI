
import sklearn.datasets as sd
import sklearn.utils as su
import  sklearn.tree as st
import  sklearn.metrics as sm
import  sklearn.ensemble as se

#读取数据
boston=sd.load_boston()
# print(boston.feature_names)
# print(boston.data[0],boston.data.shape)
# print(boston.target[0],boston.target.shape)

#打乱数据集后，拆分训练集与测试集
x,y=su.shuffle(boston.data,boston.target,random_state=7)
# print(x,y)
train_size=int(len(x)*0.8)
# print(train_size)
train_x,test_x,train_y,test_y=x[:train_size],x[train_size:],y[:train_size],y[train_size:]
# print(train_x,test_x,train_y,test_y)

#构建决策树回归器模型，训练模型
model=st.DecisionTreeRegressor(max_depth=5)

#构建正向激励回归器
model=se.AdaBoostRegressor(model,n_estimators=400,random_state=7)
model.fit(train_x,train_y)
pred_test_y=model.predict(test_x)
pred_train_y=model.predict(train_x)
print(sm.r2_score(test_y,pred_test_y))
print(sm.r2_score(train_y,pred_train_y))
#输出绝对误差
print(sm.mean_absolute_error(test_y,pred_test_y))

#特征重要性
fi=model.feature_importances_
print(fi)
















