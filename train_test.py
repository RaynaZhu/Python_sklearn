# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

# Multinomial Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf1 = MultinomialNB(alpha=0.01)

# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier()

# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
clf3 = LogisticRegression(penalty='l2')

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier(n_estimators=8)

# Decision Tree Classifier
from sklearn import tree
clf5 = tree.DecisionTreeClassifier()

# SVM Classifier
from sklearn.svm import SVC
clf6 = SVC(kernel='rbf', probability=True)

# GBDT(Gradient Boosting Decision Tree) Classifier
from sklearn.ensemble import GradientBoostingClassifier
clf7 = GradientBoostingClassifier(n_estimators=200)

# Voting Classifier
from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[('NB', clf1), ('KNN', clf2), ('LR', clf3), ('RF', clf4),
                                    ('DT', clf5), ('SVM', clf6), ('GBDT', clf7)], voting='hard')


model = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, eclf]
classifiers = ['naive_bayes_classifier',
                'knn_classifier',
                'logistic_regression_classifier',
                'random_forest_classifier',
                'decision_tree_classifier',
                'svm_classifier',
                'gradient_boosting_classifier',
                'Ensemble']
# load data
print('reading data...')
'''iris = datasets.load_iris()
X, y = iris.data, iris.target
'''
data_file = r'/Users/raynazhu/Documents/MATLAB/data/data1.mat'
data = scipy.io.loadmat(data_file)
X = data['testdata']
Y = data['target']
y = np.array(list(itertools.chain.from_iterable(Y))) # 使Y变成一维数组

# 特征归一化到【0，1】
X = MinMaxScaler().fit_transform(X)


num_sample, num_feat = X.shape
print('******************** Data Info *********************')
print('#The number of sample is : %d, #the number of feature is : %d' % (num_sample, num_feat))

for clf, classifier in zip(model, classifiers):
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), classifier))