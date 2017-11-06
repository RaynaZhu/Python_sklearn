from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

# method 1: pickle
import pickle
# save
with open('clf.pickle', 'wb') as f:
    pickle.dump(clf, f)
# restore
with open('clf.pickle', 'rb')as f:
    clf2 = pickle.load(f)

# method 2: joblib
from sklearn.externals import joblib
# save
joblib.dump(clf, 'clf.pickle')
# restore
clf3 = joblib.load()
