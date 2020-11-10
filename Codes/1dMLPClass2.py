#1e MLPClassifier optimizer
# import necessary packages
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

# ensure the same random numbers appear every time
np.random.seed(0)

# display images in notebook
#matplotlib inline
plt.rcParams['figure.figsize'] = (12,12)


# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target

print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
print("labels = (n_inputs) = " + str(labels.shape))


# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)
print("X = (n_inputs, n_features) = " + str(inputs.shape))


# choose some random images to display
indices = np.arange(n_inputs)
random_indices = np.random.choice(indices, size=5)

for i, image in enumerate(digits.images[random_indices]):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Label: %d" % digits.target[random_indices[i]])
plt.show()



from sklearn.model_selection import train_test_split

# one-liner from scikit-learn library
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
                                                    test_size=test_size)

max_iter=5000 #default=200

score = 0
etalist = np.logspace(-1,-6, num=20)
reglist= np.logspace(0,-10, num=10)
optimal = [0,0]
for eta in tqdm(etalist):
    for lmbd in reglist:
        clf = MLPClassifier(activation='logistic',solver='sgd',alpha=lmbd,learning_rate_init=eta,
                                            max_iter=max_iter).fit(X_train, Y_train)
        #clf.predict_proba(X_test[:1])
        #predict = clf.predict(X_test[:5, :])

    if score < clf.score(X_test, Y_test):
        score = clf.score(X_test, Y_test)
        optimal[0] = eta; optimal[1] = lmbd


print('optimal score = ', score)
print('optimal learning rate = ', optimal[0])
print('optimal lambda =', optimal[1])
