import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import numpy

#Draw your own digit in d with 16 being black and 0 being white
d = numpy.array(
    [[16, 16, 16, 16, 16, 16, 16, 0],
    [0, 0, 0, 0, 0, 16, 16, 0],
    [0, 0, 0, 0, 0, 16, 16, 0],
    [0, 0, 0, 0, 0, 16, 0, 0],
    [0, 0, 0, 0, 0, 16, 0, 0],
    [0, 0, 0, 0, 0, 16, 0, 0],
    [0, 0, 0, 0, 16, 0, 0, 0],
    [0, 0, 0, 0, 16, 0, 0, 0]])

plt.imshow(d, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

dd = d.reshape(1, -1)

clf = svm.SVC(gamma=0.001, C=100)

digits = datasets.load_digits()

x,y = digits.data[:-10], digits.target[:-10]
clf.fit(x,y)

print('Prediction:',clf.predict(dd))
