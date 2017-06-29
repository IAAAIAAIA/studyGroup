import keras
import numpy as np
import progressbar
import time
from Matteos_Data_Loading_Methods import Data_Generators
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

class NeuralNetwork:
    def __init__(self, num_layers, category):
        self.category = category
        self.model = Sequential()
        for i in range(num_layers):
            #little bit of math should make the number of filters have a quadratic relationship
            self.model.add(Conv2D((int)(-(i-num_layers/2)**2 + num_layers**2/4)+32, (3,3), strides= (1,1),
                                        padding='same', input_shape=(256, 256, 3)))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling2D(pool_size=(3, 3)))
        self.model.add(Flatten())
        self.model.add(Dense(1))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        
    def train(self, X_path, y_path, train_decimal=.7, val_decimal=.2, test_decimal=.1, batch_size=500):
        data = Data_Generators(X_path, y_path, val_decimal=val_decimal, 
                               train_decimal=train_decimal, test_decimal=test_decimal)
        bar = progressbar.ProgressBar();
        for batch_X, batch_y in bar(zip(data.X_train(), data.y_train('primary'))):
            batch_X = np.array(batch_X)
            batch_y = np.array(batch_y)
            self.model.fit(batch_X, batch_y, batch_size=8, epochs=1, verbose=0, 
                      validation_data=(data.X_val(), data.y_val('primary')))
            time.sleep(0.02)
        score = self.model.evaluate(data.X_test(), data.y_test('primary'), verbose=0)
        return (score)
        