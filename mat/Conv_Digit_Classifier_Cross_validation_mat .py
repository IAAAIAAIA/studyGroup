'''

#%matplotlib inline
import theano
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15,20)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.utils import np_utils

nb_classes = 10

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("X_test original shape", X_test.shape)
    print("y_test original shape", y_test.shape)
    print("X_train original shape", X_train.shape)
    print("y_train original shape", y_train.shape)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    X_reshaped_train = X_train.reshape(60000, 1, 28, 28)
    X_reshaped_test = X_test.reshape(10000, 1, 28, 28)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test, X_reshaped_train, X_reshaped_test

def create_model():
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu", input_shape=(1, 28, 28)))
    model.add(BatchNormalization()) # epsilon=1e-06, mode=0, momentum=0.99, weights=None))
    #model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'))
    model.add(Dropout(0.25))
    
    #model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu"))
    ##model.add(Convolution2D(32, 3, 3, border_mode="same", activation="relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'))
    ##model.add(MaxPooling2D(pool_size=(1, 1)))
    #model.add(Dropout(0.2))
    
    model.add(Convolution2D(32, 3, 3, border_mode="same", activation="relu"))
    model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu"))
    # model.add(Convolution2D(2, 1, 1, border_mode="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.1))
    model.add(Flatten())

    model.add(Dense(156))
    model.add(BatchNormalization()) # epsilon=1e-06, mode=0, momentum=0.99, weights=None))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="adam")

    return model

if __name__ == "__main__":
    fold = 0
    nsplits = 4 # 7
    X_train, Y_train, X_test, Y_test, X_reshaped_train, X_reshaped_test = load_data()
    
    skf = StratifiedKFold(n_splits=nsplits, shuffle=True)
    skf.get_n_splits(X_train, Y_train)

    ###
    #model = None
    #model = create_model()
    ###
    model = None
    model = load_model('partly_trained_5.h5')
    ###
    #model.fit(X_reshaped_train, Y_train, verbose=1, batch_size=128, nb_epoch=1)

    score_actual_test = model.evaluate(X_reshaped_test, Y_test, verbose=0)
    print('Actual Test score: ', score_actual_test)

    model.save('partly_trained_6.h5')
    del model

    print(skf)
    for train_index, test_index in skf.split(np.zeros(60000), np.zeros(60000)):
        fold += 1
        print("Fold: ", fold, "/", nsplits)
        X_total_train, X_total_test = X_train[train_index], X_train[test_index]
        Y_total_train, Y_total_test = Y_train[train_index], Y_train[test_index]
        X_total_train = X_total_train.reshape(X_total_train.shape[0], 1, 28, 28)
        X_total_test = X_total_test.reshape(X_total_test.shape[0], 1, 28, 28)
        model = load_model('partly_trained_6.h5')

        model.fit(X_total_train, Y_total_train, verbose=1, batch_size=128, nb_epoch=3) # nb_epoch=15
        score_fold = model.evaluate(X_total_test, Y_total_test, verbose=0)
        score_actual_test = model.evaluate(X_reshaped_test, Y_test, verbose=0)
        print('Fold test score: ', score_fold)
        print('Actual Test score: ', score_actual_test)

        model.save('partly_trained_6.h5')
        del model

    model = load_model('partly_trained_6.h5')    
    model.fit(X_reshaped_train, Y_train, verbose=1, batch_size=128, nb_epoch=1)
    score_fold = model.evaluate(X_reshaped_train, Y_train, verbose=0)
    score_actual_test = model.evaluate(X_reshaped_test, Y_test, verbose=0)
    print('Fold test score: ', score_fold)
    print('Actual Test score: ', score_actual_test)


model = load_model('partly_trained_6.h5')
score_test = model.evaluate(X_reshaped_test, Y_test, verbose=0)
print('Test score (the one that matters): ', score_test)
score_train = model.evaluate(X_reshaped_train, Y_train, verbose=0)
print('Train score: ', score_train)

predicted_classes = model.predict_classes(X_reshaped_test)


# partly_trained_5:
# Test score (the one that matters):  0.0154483271036
# Train score:  0.00184156776947

'''
# TO TEST

import theano
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15,20)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.utils import np_utils

nb_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


model = load_model('partly_trained_5.h5')

X_train = X_train.reshape(60000, 1, 28, 28)
X_test = X_test.reshape(10000, 1, 28, 28)

score_test = model.evaluate(X_test, Y_test, verbose=0)
print('Test score (the one that matters): ', score_test)
score_train = model.evaluate(X_train, Y_train, verbose=0)
print('Train score: ', score_train)
predicted_classes = model.predict_classes(X_test)

#'''

'''

#%matplotlib inline
import theano
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15,20)
from keras.datasets import mnist
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.utils import np_utils

nb_classes = 10

def load_data():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    #print("X_test original shape", X_test.shape)
    #print("y_test original shape", y_test.shape)
    #print("X_train original shape", X_train.shape)
    #print("y_train original shape", y_train.shape)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    X_reshaped_train = X_train.reshape(60000, 1, 28, 28)
    X_reshaped_test = X_test.reshape(10000, 1, 28, 28)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test, X_reshaped_train, X_reshaped_test

def create_model():
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu", input_shape=(1, 28, 28)))
    model.add(BatchNormalization()) # epsilon=1e-06, mode=0, momentum=0.99, weights=None))
    model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'))
    model.add(Dropout(0.2))

    #model.add(Convolution2D(64, 3, 3, border_mode="same", activation="relu"))
    #model.add(Convolution2D(32, 3, 3, border_mode="same", activation="relu"))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th'))
    #model.add(MaxPooling2D(pool_size=(1, 1)))
    #model.add(Dropout(0.2))

    model.add(Convolution2D(32, 3, 3, border_mode="same", activation="relu"))
    model.add(Convolution2D(64, 5, 5, border_mode="same", activation="relu"))
    # model.add(Convolution2D(2, 1, 1, border_mode="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.1))
    model.add(Flatten())

    model.add(Dense(792))
    #model.add(BatchNormalization()) # epsilon=1e-06, mode=0, momentum=0.99, weights=None))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    nad = optimizers.Nadam(lr=0.008, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    model.compile(loss="categorical_crossentropy", optimizer=nad)

    return model

if __name__ == "__main__":
    fold = 0
    nsplits = 6 # 7
    X_train, Y_train, X_test, Y_test, X_reshaped_train, X_reshaped_test = load_data()
    print("X_train shape", X_train.shape)
    
    skf = StratifiedKFold(n_splits=nsplits, shuffle=True)
    skf.get_n_splits(X_train, Y_train)

    print("X_total_train shape", X_reshaped_train.shape)

    ###
    #model = None
    #model = create_model()
    ###
    model = None
    model = load_model('partly_trained_5.h5')
    ###
    model.fit(X_reshaped_train, Y_train, verbose=1, batch_size=128, nb_epoch=1)

    model.save('partly_trained_6.h5')
    del model

    print(skf)
    for train_index, test_index in skf.split(np.zeros(60000), np.zeros(60000)):
        fold += 1
        print("Fold: ", fold, "/", nsplits)
        X_total_train, X_total_test = X_train[train_index], X_train[test_index]
        Y_total_train, Y_total_test = Y_train[train_index], Y_train[test_index]
        X_total_train = X_total_train.reshape(X_total_train.shape[0], 1, 28, 28)
        X_total_test = X_total_test.reshape(X_total_test.shape[0], 1, 28, 28)
        model = load_model('partly_trained_6.h5')

        model.fit(X_total_train, Y_total_train, verbose=1, batch_size=128, nb_epoch=8) # nb_epoch=15
        score_fold = model.evaluate(X_total_test, Y_total_test, verbose=0)
        score_actual_test = model.evaluate(X_reshaped_test, Y_test, verbose=0)
        print('Fold test score: ', score_fold)
        print('Actual Test score: ', score_actual_test)

        model.save('partly_trained_6.h5')
        del model

    model = load_model('partly_trained_6.h5')
    model.fit(X_reshaped_train, Y_train, verbose=1, batch_size=128, nb_epoch=1)
    score_fold = model.evaluate(X_reshaped_train, Y_train, verbose=0)
    score_actual_test = model.evaluate(X_reshaped_test, Y_test, verbose=0)
    print('Fold test score: ', score_fold)
    print('Actual Test score: ', score_actual_test)
    model.save('partly_trained_6.h5')
    del model

model = load_model('partly_trained_6.h5')
score_test = model.evaluate(X_reshaped_test, Y_test, verbose=0)
print('Test score (the one that matters): ', score_test)
score_train = model.evaluate(X_reshaped_train, Y_train, verbose=0)
print('Train score: ', score_train)

predicted_classes = model.predict_classes(X_reshaped_test)

# partly_trained_3:
# Fold test score:  0.0091460001624
# Test score (the one that matters):  0.0277665542863
# Train score:  0.00428307721489

# partly_trained_4:
# Fold test score:  0.0106372920116
# Actual Test score:  0.0356249902597
# Test score (the one that matters):  0.0272345396012
# Train score:  0.00996801403004


# TO TEST

import theano
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15,20)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.utils import np_utils

nb_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


model = load_model('partly_trained_4.h5')

X_train = X_train.reshape(60000, 1, 28, 28)
X_test = X_test.reshape(10000, 1, 28, 28)

score_test = model.evaluate(X_test, Y_test, verbose=0)
print('Test score (the one that matters): ', score_test)
score_train = model.evaluate(X_train, Y_train, verbose=0)
print('Train score: ', score_train)
predicted_classes = model.predict_classes(X_test)

'''