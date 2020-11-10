#1bc test code against sklearn - optimizer
# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random import random, seed
import numpy as np
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
# ensure the same random numbers appear every time
np.random.seed(0)

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
m = len(x)
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

#adding normalized noise to the Franke function
sigma2 = 0.1
z = FrankeFunction(x, y) + np.random.normal(0,sigma2, len(x))

x = np.ravel(x)
y = np.ravel(y)
z = np.ravel(z)

#  The design matrix now as function of a given polynomial
def create_X(x, y, n ):
        if len(x.shape) > 1:
                x = np.ravel(x)
                y = np.ravel(y)

        N = len(x)
        l = int((n+1)*(n+2)/2)          # Number of elements in beta
        X = np.ones((N,l))
        idx = 0
        for i in range(1,n+1):
                q = int((i)*(i+1)/2)
                for k in range(i+1):
                        X[:,idx] = (x**(i-k))*(y**k)
                        idx +=1

        return X
X = create_X(x, z, n=7)
# We split the data in test and training data
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train[:,0] = 1
X_test[:,0] = 1


# one-liner from scikit-learn library
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(X, z, train_size=train_size,
                                                    test_size=test_size)

max_iter=5000 #default=200

score = 0
etalist = np.logspace(-1,-6, num=20)
reglist= np.logspace(0,-10, num=20)
optimal = [0,0]
for eta in tqdm(etalist):
    for lmbd in reglist:
        regr = MLPRegressor(activation='relu',solver='sgd',alpha=lmbd,learning_rate_init=eta,
                                            max_iter=max_iter).fit(X_train, Y_train)
        #regr.predict(X_test)

    if score < regr.score(X_test, Y_test):
        score = regr.score(X_test, Y_test)
        optimal[0] = eta; optimal[1] = lmbd


print('optimal score = ', score)
print('optimal learning rate = ', optimal[0])
print('optimal lambda =', optimal[1])


'''
PS C:\python\project2> python .\1bcMLPReg2.py
100%|█████████████████████████████████████████████████████████████████████████████████████████| 20/20 [01:24<00:00,  4.24s/it]
optimal score =  0.9670078678439595
optimal learning rate =  0.0545559478116852
optimal lambda = 1e-10
'''

''' relu
PS C:\python\project2> python .\1bcMLPReg2.py
100%|█████████████████████████████████████████████████████████████████████████████████████████| 20/20 [01:53<00:00,  5.67s/it]
optimal score =  0.992733368588985
optimal learning rate =  0.0545559478116852
optimal lambda = 1e-10
PS C:\python\project2>
'''
