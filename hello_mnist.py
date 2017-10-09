from sklearn.externals import joblib
import scipy.ndimage
from PIL import Image
import numpy as np
import glob,os
import numpy as np
from random import randint
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
initialTime = datetime.now()
print(initialTime)
mnist = fetch_mldata('MNIST original')


#looks like X and y are values and "data" and "target" were keys to a dict
X = mnist["data"]
y = mnist["target"]

print("This is the mnist data")
print(X)

print("This is the mnist label")
print(y)


#split the data to training,validation, and testing.  I arbitrarily had a .75, .125, .125 split
X_train, X_valid, X_test = X[:60000],X[60000:70000], X[70000:]

y_train, y_valid, y_test = y[:60000],y[60000:70000], y[70000:]

#we dont actually need X_valid and y_valid if we do cross validation
X_test = X[60000:]
y_test = y[60000:]

print("before shuffle")
#shuffle all the training data
shuffle_idx = np.random.permutation(60000)
X_train = X[shuffle_idx]
y_train = y[shuffle_idx]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.fit_transform(X_test.astype(np.float32))
"""
mnist_svm = joblib.load('my_first_big_model_svm_mnist.sav')
y_pred_test = mnist_svm.predict(X_test_scaled)
print(y_pred_test)
print(accuracy_score(y_test, y_pred_test))
"""
temp=(Image.open('6.jpg'))
temp.show()
somePic = np.array(temp)
somePic = somePic[:,:,:1].reshape(784)
someIdx = randint(0,60000)



yetAnotherIdx = randint(0,60000)
#print(somePic)
X = mnist["data"]
anElement = X[someIdx]
anotherElement = X[yetAnotherIdx]



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.fit_transform(X_test.astype(np.float32))


X_somePartition = X_test_scaled[60000:80000:1597]

print(y[60000:80000:1597])

myArray = [anElement,anotherElement]

myArray = scaler.fit_transform(np.asarray(myArray).astype(np.float32))
#print(temp)
clf = joblib.load('my_first_not_trivial_svm_model.pkl')

y_pred = clf.predict(X_somePartition)

"""
print(y_pred)
"""

"""
myArray = [somePic,anElement,anotherElement]
print("LABELS SHOULD BE")
print(y[someIdx])
print(y[yetAnotherIdx])
print("SOME PARTITION")
print(clf.predict(X_somePartition))
print(clf.predict(myArray))
"""

