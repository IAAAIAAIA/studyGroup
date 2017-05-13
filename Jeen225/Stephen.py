from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dropout
from keras.layers.core import Activation 
from keras.layers.core import Dense
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7)

nb_classes = 10

(X_train,y_train),(X_test,y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i],cmap='gray',interpolation='none')
    plt.title("Class {}".format(y_train[i]))
Y_train = np_utils.to_categorical(y_train,nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=128, epochs=4,verbose = 1,validation_data=(X_test, Y_test))
score = model.evaluate(X_test,Y_test,verbose=0)
##print('Test score:', score[0])
##print('Test accuracy:',score[1])
print(score)