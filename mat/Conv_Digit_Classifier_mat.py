#%matplotlib inline

from keras.layers.normalization import BatchNormalization

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15,20) # Make the figures a bit bigger

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.utils import np_utils

nb_classes = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_test original shape", X_test.shape)
print("y_test original shape", y_test.shape)
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))
    plt.show

X_train = X_train.reshape(60000, 1, 784, 1)
X_test = X_test.reshape(10000, 1, 784, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print("Training matrix shape", Y_train.shape)
print("Testing matrix shape", Y_test.shape)

model = Sequential()
model.add(Convolution2D(392, 3, 1,border_mode="same", activation="relu",input_shape=(1, 784, 1)))

model.add(BatchNormalization())#epsilon=1e-06, mode=0, momentum=0.9, weights=None

#model.add(Convolution2D(392, 3, 1,border_mode="same", activation="relu"))
model.add(Convolution2D(392, 3, 1, border_mode="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(1,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(392))
model.add(BatchNormalization())#epsilon=1e-06, mode=0, momentum=0.9, weights=None
model.add(Activation('relu'))
model.add(Dropout(0.1))
#cnn.add(Dense(784, activation="relu"))
#cnn.add(Dropout(0.2))

#cnn.add(Convolution2D(784, 3, 1, border_mode="same", activation="relu"))
#cnn.add(Dense(784, activation="relu"))
#cnn.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adam")

model.fit(X_train, Y_train,verbose=1,batch_size=128,nb_epoch=4)

score = cnn.evaluate(X_test, Y_test, verbose=0)
print('Test score: ', score)

# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = model.predict_classes(X_test)

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

predicted_classes_train = model.predict_classes(X_train)


correct_indices_train = np.nonzero(predicted_classes_train == y_train)[0]
incorrect_indices_train = np.nonzero(predicted_classes_train != y_train)[0]