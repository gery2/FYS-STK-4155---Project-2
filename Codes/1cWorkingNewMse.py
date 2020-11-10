#1c workingnewMSE
# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random import random, seed
import numpy as np
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm
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
#X_train[:,0] = 1
#X_test[:,0] = 1


# one-liner from scikit-learn library
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(X, z, train_size=train_size,
                                                    test_size=test_size)


print("Number of training images: " + str(len(X_train)))
print("Number of test images: " + str(len(X_test)))

# building our neural network

n_inputs, n_features = X_train.shape
n_hidden_neurons = 100
n_categories = 1 #10 with classification, 1 with Franke because we only want 1 output value

# we make the weights normally distributed using numpy.random.randn

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# setup the feed-forward pass, subscript h = hidden layer
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def deriv_sigmoid(x):
    x = 1/(1 + np.exp(-x))
    return x * (1 - x)

def RELU(x):
    return np.maximum(0, x)

def deriv_RELU(x):
    return np.where(x < 0, 0, 1)

def leaky_RELU(x):
    return np.maximum(0.01, x)

def deriv_leaky_RELU(x):
    return np.where(x < 0, 0.01, 1)

def feed_forward(X):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, hidden_weights) + hidden_bias
    # activation in the hidden layer
    a_h = sigmoid(z_h); d_a_h = deriv_sigmoid(z_h)
    #a_h = RELU(z_h); d_a_h = deriv_RELU(z_h)
    #a_h = leaky_RELU(z_h); d_a_h = deriv_leaky_RELU(z_h)
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias

    return z_h,a_h,z_o


# weights and bias in the hidden layer
hidden_weights = np.random.randn(n_features, n_hidden_neurons)
hidden_bias = np.zeros(n_hidden_neurons) + 0.0

# weights and bias in the output layer
output_weights = np.random.randn(n_hidden_neurons, n_categories)
output_bias = np.zeros(n_categories) + 0.0
print('-----------', hidden_weights.shape)




# to categorical turns our integer vector into a onehot representation
from sklearn.metrics import accuracy_score

def feed_forward_train(X):
    # weighted sum of inputs to the hidden layer
    z_h = np.matmul(X, hidden_weights) + hidden_bias
    # activation in the hidden layer
    a_h = sigmoid(z_h); d_a_h = deriv_sigmoid(z_h)
    #a_h = RELU(z_h); d_a_h = deriv_RELU(z_h)
    #a_h = leaky_RELU(z_h); d_a_h = deriv_leaky_RELU(z_h)
    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias

    return a_h, z_o, d_a_h

# we obtain a prediction by taking the class with the highest likelihood
def predict(X):
    a_h, z_o, d_a_h = feed_forward_train(X)
    return z_o

predictions = predict(X_train)
print("predictions = (n_inputs) = " + str(predictions.shape))
print("prediction for image 0: " + str(predictions[0]))
print("correct label for image 0: " + str(Y_train[0]))

def backpropagation(X, Y, ):
    a_h, z_o, d_a_h = feed_forward_train(X)
    # error in the output layer
    error_output = z_o - Y.reshape(-1,1)
    # error in the hidden layer
    error_hidden = np.matmul(error_output, output_weights.T) * d_a_h

    # gradients for the output layer
    output_weights_gradient = np.matmul(a_h.T, error_output)
    output_bias_gradient = np.sum(error_output, axis=0)

    # gradient for the hidden layer
    hidden_weights_gradient = np.matmul(X.T, error_hidden)
    hidden_bias_gradient = np.sum(error_hidden, axis=0)

    return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

print("Old MSE on training data: " + str(MSE(predict(X_train), Y_train)))
print("Old R2 on training data: " + str(R2(Y_train,predict(X_train))))

new_MSE = 10**3
new_R2 = 10**3
etalist = np.logspace(-3.2,-7, num=30)
reglist= np.logspace(0,-10, num=30)
optimal = [0,0]

for eta in tqdm(etalist):
    for lmbd in reglist:
        for i in range(10):
            # calculate gradients
            dWo, dBo, dWh, dBh = backpropagation(X_train, Y_train)

            # regularization term gradients
            dWo += lmbd * output_weights
            dWh += lmbd * hidden_weights

            # update weights and biases
            output_weights -= eta * dWo
            output_bias -= eta * dBo
            hidden_weights -= eta * dWh
            hidden_bias -= eta * dBh
            if new_MSE > MSE(predict(X_train), Y_train):
                new_MSE = MSE(predict(X_train), Y_train)
                optimal[0] = eta; optimal[1] = lmbd
            if new_R2 > R2(Y_train,predict(X_train)):
                new_R2 = R2(Y_train,predict(X_train))

#output_weights,output_bias,hidden_weights,hidden_bias
print("New MSE on training data: ", new_MSE)
print("New R2 on training data: ", new_R2)



print('optimal learning rate = ', optimal[0])
print('optimal lambda =', optimal[1])




''' (sigmoid)
PS C:\python\project2> python .\1cWorkingNewMse.py
Number of training images: 320
Number of test images: 80
predictions = (n_inputs) = (320, 1)
prediction for image 0: [3.54291735]
correct label for image 0: 0.7857906119686391
Old MSE on training data: 2915.651640546522
Old R2 on training data: -30310.805928311907
100%|█████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:38<00:00,  1.30s/it]
New MSE on training data:  34.08069243500943
New R2 on training data:  -48117116.8297108
optimal learning rate =  0.000630957344480193
optimal lambda = 0.041753189365604
'''

''' (RELU)
PS C:\python\project2> python .\1cWorkingNewMse.py
Number of training images: 320
Number of test images: 80
predictions = (n_inputs) = (320, 1)
prediction for image 0: [0.65367285]
correct label for image 0: 0.7857906119686391
Old MSE on training data: 7655.356073950914
Old R2 on training data: -79585.89728181242
100%|█████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:26<00:00,  1.14it/s]
New MSE on training data:  30.78036746413171
New R2 on training data:  -3.427543720545378e+54
optimal learning rate =  0.00046662232927030696
optimal lambda = 0.0007880462815669905
'''

''' (leaky RELU)
PS C:\python\project2> python .\1cWorkingNewMse.py
Number of training images: 320
Number of test images: 80
predictions = (n_inputs) = (320, 1)
prediction for image 0: [0.72614661]
correct label for image 0: 0.7857906119686391
Old MSE on training data: 7846.810257331812
Old R2 on training data: -81576.30037733361
100%|█████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:33<00:00,  1.11s/it]
New MSE on training data:  71.48933804475568
New R2 on training data:  -681177486.4585073
optimal learning rate =  1e-07
optimal lambda = 1e-10
'''
