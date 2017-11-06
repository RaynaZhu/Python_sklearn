
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io
import numpy as np

# Load data
datafile = r'/Users/raynazhu/Documents/MATLAB/data/testdata.mat'

data = scipy.io.loadmat(datafile)
X = data['data']
H = data['all_human']
for i in range(24):
    h = H[:, i]
    #h = X[:500, i]
    b = X[500:700, i]
    #b2 = X[600:700, i]
    f = plt.figure(i)
    sns.set(style="white", palette="muted", color_codes=True)
    h = sns.kdeplot(h, label=r'human', color='R', shade=True)
    b = sns.kdeplot(b, label=r'bot', color='B', shade=True)
    #sns.kdeplot(b2, label=r'bot2', color='G', shade=True)
    plt.title('Probability density distribution')
    plt.show()
    f.savefig("%d.png"%(i+1))
