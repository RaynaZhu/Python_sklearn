from sklearn import datasets
import scipy.io
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
datafile = r'/Users/raynazhu/Documents/MATLAB/data/data.mat'
data = scipy.io.loadmat(datafile)
X = data['data'][1:20, 1:3]
Y = data['target']
y = np.array(list(itertools.chain.from_iterable(Y)))


'''

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
'''
# print(Y)
print(type(Y), Y.shape)
print(y)
print(type(y), y.shape)

