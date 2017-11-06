# -*-coding:utf-8 -*-


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from sklearn.datasets import load_iris

iris = load_iris()
# print iris#iris的４个属性是：萼片宽度　萼片长度　花瓣宽度　花瓣长度　标签是花的种类：setosa versicolour virginica
print(iris['target'].shape)
rf = RandomForestClassifier()  # 这里使用了默认的参数设置
rf.fit(iris.data[:150], iris.target[:150])  # 进行模型的训练
#
# 随机挑选两个预测不相同的样本
print(rf.feature_importances_)
tx = iris.data[141:150]
ty = iris.target[141:150]
print(rf.predict_proba(tx))
print(rf.score(tx, ty, sample_weight=None))
#print('instance 1 prediction；', rf.predict(instance[1]))
#print(iris.target[100], iris.target[109])

'''from sklearn.cross_validation import cross_val_score, ShuffleSplit
X = iris["data"]
Y = iris["target"]
names = iris["feature_names"]
rf = RandomForestRegressor()
scores = []
for i in range(X.shape[1]):
     score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",
                              cv=ShuffleSplit(len(X), 3, .3))
     scores.append((round(np.mean(score), 3), names[i]))
print(sorted(scores, reverse=True))'''