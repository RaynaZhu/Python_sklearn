# -*-coding:utf-8-*-

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4, test_size=0.3)

# print(y_train)

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel("Value of k for knn")
plt.ylabel("Cross-Validated Accuracy")
plt.show()
'''
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)
print(y_test)
acc = 0
count = 0
for i in range(y_test.size):
    if y_test[i] == y_pred[i]:
        count = count + 1
    acc = (count/y_test.size)*100

print("Accuracy: %0.3f" % acc)
print("%0.3f" % knn.score(X_test, y_test))
'''

