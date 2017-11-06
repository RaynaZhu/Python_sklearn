# -*-coding:utf-8-*-
"""
=========================================
特征选择：1单变量特征选择，2基于树的选择，3带交叉验证的递归特征消除
=========================================
根据三种特征选择方法，综合选取所需特征。
"""
print(__doc__)
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load  data
datafile = r'/Users/raynazhu/Documents/MATLAB/data/data1.mat'
data = scipy.io.loadmat(datafile)
X = data['testdata']
y = data['target']
y = np.ravel(y)

# 特征归一化到【0，1】
X = MinMaxScaler().fit_transform(X)

'''
# 1 Univariate feature selection
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, mutual_info_classif

plt.figure(1)
plt.clf()

X_indices = np.arange(X.shape[-1])

# 1.1 Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
selector1 = SelectPercentile(f_classif, percentile=10)
selector1.fit(X, y)
#scores = -np.log10(selector.pvalues_)
scores1 = selector1.scores_
scores1 /= scores1.max()
plt.bar(X_indices - .45, scores1, width=.2,
        label=r'f_classif', color='c')

# 1.2Univariate feature selection with Chi-squared
selector2 = SelectPercentile(chi2, percentile=10)
selector2.fit(X, y)
scores2 = selector2.scores_
scores2 /= scores2.max()
plt.bar(X_indices - .25, scores2, width=.2,
        label=r'chi2', color='navy')

# 1.3Univariate feature selection with mutual_info_classif
selector3 = SelectPercentile(mutual_info_classif, percentile=10)
selector3.fit(X, y)
scores3 = selector3.scores_
scores3 /= scores3.max()
plt.bar(X_indices - .05, scores3, width=.2,
        label=r'mutual_info_classif', color='darkorange')

plt.title("The results of Univariate feature selection")
plt.xlabel('Feature number')
plt.yticks(())
plt.axis('tight')
plt.xticks(range(X.shape[1]))
plt.legend(loc='upper right')
plt.show()

# 2 Feature importances with forests of trees
from sklearn.ensemble import ExtraTreesClassifier

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(2)
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        label=r'Feature importances with forests of trees', color="c")
plt.xlabel("Feature number")
plt.axis('tight')
plt.legend(loc='upper right')
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
'''
# 3 Feature selection uses Recursive feature elimination with cross-validation.
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
plt.figure(3)
plt.clf()

# 3.1 Feature selection uses RFE with the logistic regression algorithm
lr = LogisticRegression()
rfecv1 = RFECV(estimator=lr, step=1, cv=StratifiedKFold(5),
              scoring='accuracy')
rfecv1.fit(X, y)

print("Optimal number of features : %d" % rfecv1.n_features_)

# Plot number of features VS. cross-validation scores
plt.plot(range(1, len(rfecv1.grid_scores_) + 1), rfecv1.grid_scores_, label="Logistic Regression", color='darkorange')

# 3.2 Feature selection uses RFE with the SVM algorithm
# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
rfecv2 = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5),
              scoring='accuracy')
rfecv2.fit(X, y)

print("Optimal number of features : %d" % rfecv2.n_features_)

# Plot number of features VS. cross-validation scores
plt.plot(range(1, len(rfecv2.grid_scores_) + 1), rfecv2.grid_scores_, label="SVM", color='c')

plt.title("Optimal number of features")
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.legend(loc='best')
plt.show()
