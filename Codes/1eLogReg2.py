#1eLogisticRegression
from sklearn import datasets
from tqdm import tqdm
# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target
print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
print("labels = (n_inputs) = " + str(labels.shape))

import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize = 20)
plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression

score = 0
etalist = np.logspace(0,-6, num=10)
reglist= np.logspace(0,-10, num=10)
optimal = [0,0]
for eta in tqdm(etalist):
    for lmbd in reglist:
        logisticRegr = LogisticRegression(max_iter=1000,multi_class='auto',solver='liblinear')
        logisticRegr.fit(x_train, y_train)
        logisticRegr.predict(x_test[0].reshape(1,-1))
        logisticRegr.predict(x_test[0:10])
        predictions = logisticRegr.predict(x_test)
    if score < logisticRegr.score(x_test, y_test):
        score = logisticRegr.score(x_test, y_test)
        optimal[0] = eta; optimal[1] = lmbd


print('optimal score = ', score)
print('optimal learning rate = ', optimal[0])
print('optimal lambda =', optimal[1])

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

plt.figure(figsize=(9,9))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], rotation=45, size = 10)
plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size = 10)
plt.tight_layout()
plt.ylabel('Actual label', size = 15)
plt.xlabel('Predicted label', size = 15)
width, height = cm.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(cm[x][y]), xy=(y, x),
        horizontalalignment='center',
        verticalalignment='center')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
index = 0
misclassifiedIndexes = []
for i in range(len(y_test)):
        if y_test[i] != predictions[i]:
            misclassifiedIndexes.append(i)

plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    plt.imshow(np.reshape(x_test[badIndex], (8,8)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(predictions[badIndex], y_test[badIndex]), fontsize = 15)
plt.show()
