#project1dcrossval minibatches
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from math import exp, sqrt
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm
from numba import jit


seed(140)
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

#legger til normalfordelt støy til funksjonen
sigma2 = 0.3
z = (FrankeFunction(x, y) + np.random.normal(0,sigma2, len(x))).reshape(-1,1)

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def create_X(x, y, n ,intercept=True):
        if len(x.shape) > 1:
                x = np.ravel(x)
                y = np.ravel(y)

        N = len(x)
        l = int((n+1)*(n+2)/2)          # Number of elements in beta
        X = np.ones((N,l))
        idx = 0
        for i in range(2-intercept,n+1):
                q = int((i)*(i+1)/2)
                for k in range(i+1):
                        X[:,idx] = (x**(i-k))*(y**k)
                        idx +=1
        return X


X = create_X(x, y, n=7)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled[:,0] = 1
X_test_scaled[:,0] = 1


I = np.eye(X.shape[1],X.shape[1])
nlambdas = 50
lambdas = np.logspace(-9, 0, nlambdas)
k = 5
kfold = KFold(n_splits = k)
train_MSE_KFold = np.zeros(nlambdas)
test_MSE_KFold = np.zeros(nlambdas)
MSE_test_CV = np.zeros(k)
MSE_train_CV = np.zeros(k)


MSE_train = 10**3
MSE_test = 10**3
train_indexes = [0,0]; test_indexes = [0,0]
t0, t1 = 0.5, 50
@jit(nopython=True)
def learning_schedule(t):
    return t0/(t+t1)

batch_size = np.linspace(3,20,18)
n_epochs = np.linspace(51,100,50)
@jit(nopython=True)
def minibatch_gradient_descent(X,z,batch_size,n_epochs):
    m = len(z)
    theta = np.random.randn(X.shape[1])
    for epoch in range(int(n_epochs)):
        indices = np.random.permutation(m)
        X = X[indices]
        z = z[indices]
        for i in range(0,m,int(batch_size)):
            X_i = X[i:i+int(batch_size)]
            z_i = z[i:i+int(batch_size)].ravel()
            #print(X_i.shape, z_i.shape, theta.shape)
            #gradients = 2*(X_i.T@(X_i@theta - z_i)) + lmb*2*theta
            gradients = 2 * X_i.T @ ((X_i @ theta)-z_i)# + lmb*2*theta
            eta = learning_schedule(epoch*m+i)
            theta = theta - eta*gradients
    return theta

for i in tqdm(batch_size):
    for j in n_epochs:
        for l in range(nlambdas):
            lmb = lambdas[l]
            q = 0
            for train_inds, test_inds in kfold.split(X_train_scaled):
                x_cv_train = X_train_scaled[train_inds]
                z_cv_train = z_train[train_inds]

                x_val = X_train_scaled[test_inds]
                z_val = z_train[test_inds]

                t0 = np.mean(z_cv_train)

                theta = minibatch_gradient_descent(x_cv_train,z_cv_train,i,j)

                ztilde = x_cv_train @ theta + t0
                MSE_train_CV[q] = MSE(z_cv_train,ztilde)
                zpredict = x_val @ theta + t0
                MSE_test_CV[q] = MSE(z_val,zpredict)

                if MSE(z_cv_train,ztilde) < MSE_train:
                    MSE_train = MSE(z_cv_train,ztilde)
                    train_indexes[0] = i; train_indexes[1] = j
                if MSE(z_val,zpredict) < MSE_test:
                    MSE_test = MSE(z_val,zpredict)
                    test_indexes[0] = i; test_indexes[1] = j;

                q += 1

            train_MSE_KFold[l] = np.mean(MSE_train_CV)
            test_MSE_KFold[l] = np.mean(MSE_test_CV)
k = (np.argmin(test_MSE_KFold))
print(lambdas[k])
fig = plt.figure()
plt.title('Ridge regression with k-fold cross-validation')
plt.xlabel('log10(lambda)')
plt.ylabel('Test Error')
plt.plot(np.log10(lambdas), (test_MSE_KFold), label='test_MSE_KFold')
plt.plot(np.log10(lambdas), (train_MSE_KFold), label='train_MSE_KFold')
plt.legend()
plt.show()


print('optimal train batch_size =', train_indexes[0])
print('optimal train n_epochs =', train_indexes[1])
print('optimal test batch_size =', test_indexes[0])
print('optimal test n_epochs =', test_indexes[1])
print('--------------------------------------')
print('MSE_train = ', MSE_train)
print('MSE_test = ', MSE_test)




'''
PS C:\python\project2> python 1acrossmini.py
100%|██████████████████████████████████████████████████████████████████████████████████| 18/18 [17:30<00:00, 58.35s/it]
2.3299518105153718e-09
optimal train batch_size = 9.0
optimal train n_epochs = 55.0
optimal test batch_size = 9.0
optimal test n_epochs = 55.0
--------------------------------------
MSE_train =  111.81346045275339
MSE_test =  21.512207182326726
'''
