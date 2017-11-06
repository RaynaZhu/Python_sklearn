import scipy.io
data_file = r'/Users/raynazhu/Documents/MATLAB/data/testdata.mat'
data = scipy.io.loadmat(data_file)
'''
X = data['data']
Y = data['target']
y = np.array(list(itertools.chain.from_iterable(Y)))

# 特征归一化到【0，1】
X = MinMaxScaler().fit_transform(X)
'''
X = data['testdata'][:, :24]
y = data['testdata'][:, 24]
print(y)