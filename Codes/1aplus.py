#1a    plus
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from math import exp, sqrt
from random import random, seed
import numpy as np
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm
seed(101)

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

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

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

batch_size = np.linspace(3,25,23)
n_epochs = np.linspace(11,100,90)
t0, t1 = 0.5, 50
def learning_schedule(t):
    return t0/(t+t1)

def minibatch_gradient_descent(X,z,batch_size,n_epochs):
    m = len(z)
    theta = np.random.randn(X.shape[1])
    for epoch in range(int(n_epochs)):
        indices = np.random.permutation(m)
        X = X[indices]
        z = z[indices]
        for i in range(0,m,int(batch_size)):
            X_i = X[i:i+int(batch_size)]
            z_i = z[i:i+int(batch_size)]
            gradients = 2 * X_i.T @ ((X_i @ theta)-z_i)
            eta = learning_schedule(epoch*m+i)
            theta = theta - eta*gradients
    return theta

def SGD(X,z):
    theta = np.random.randn(X.shape[1])
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_train[random_index:random_index+1]
            zi = z_train[random_index:random_index+1]
            gradients = 2 * xi.T @ ((xi @ theta)-zi)
            eta = learning_schedule(epoch*m+i)
            theta = theta - eta*gradients
    return theta
    
MSE_train = 10**3
MSE_test = 10**3
train_indexes = [0,0]; test_indexes = [0,0]
for i in tqdm(batch_size):
    for j in n_epochs:

        theta = minibatch_gradient_descent(X_train,z_train,i,j)
        ytildeOLS = X_train @ theta
        ypredictOLS = X_test @ theta
        if MSE(z_train,ytildeOLS) < MSE_train:
            MSE_train = (MSE(z_train,ytildeOLS))
            train_indexes[0] = i; train_indexes[1] = j
        if MSE(z_test,ypredictOLS) < MSE_test:
            MSE_test = (MSE(z_test,ypredictOLS))
            test_indexes[0] = i; test_indexes[1] = j;

print('optimal train batch_size =', train_indexes[0])
print('optimal train n_epochs =', train_indexes[1])
print('optimal test batch_size =', test_indexes[0])
print('optimal test n_epochs =', test_indexes[1])
print('--------------------------------------')
print('MSE_train = ', MSE_train)
print('MSE_test = ', MSE_test)



'''
PS C:\python\project2> python 1aplus.py
optimal train batch_size = 5.0
optimal train n_epochs = 51.0
optimal test batch_size = 5.0
optimal test n_epochs = 51.0
--------------------------------------
MSE_train =  0.008742296555417788
MSE_test =  0.006330199650607771
'''
