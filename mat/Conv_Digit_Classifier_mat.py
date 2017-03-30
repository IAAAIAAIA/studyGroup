#%matplotlib inline

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import StratifiedKFold

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15,20) # Make the figures a bit bigger

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
    X_total = np.concatenate((X_train, X_test), axis=0)
    y_total = np.concatenate((y_train, y_test), axis=0)
    print("X_total shape", X_total.shape)
    print("y_total shape", y_total.shape)

    #X_train = X_train.reshape(60000, 1, 784, 1)
    #X_test = X_test.reshape(10000, 1, 784, 1)
    #X_total = X_total.reshape(70000, 1, 784, 1)

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_total = X_total.reshape(70000, 784)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_total = X_total.astype('float32')
    X_train /= 255
    X_test /= 255 
    X_total /= 255
    #print("Training matrix shape", X_train.shape)
    #print("Testing matrix shape", X_test.shape)

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    Y_total = np_utils.to_categorical(y_total, nb_classes)
    print("Y_total shape", Y_total.shape)
    #print("Training matrix shape", Y_train.shape)
    #print("Testing matrix shape", Y_test.shape)
    return X_total, Y_total, X_train, Y_train, X_test, Y_test

def create_model():
    model = Sequential()

    model.add(Convolution2D(35, 3, 3, border_mode="same", activation="relu",input_shape=(1, 784, 1)))
    #model.add(Convolution2D(49, 3, 3, border_mode="same", activation="relu"))
    model.add(Convolution2D(35, 3, 3, border_mode="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(1,2)))
    model.add(Dropout(0.3))
    model.add(Flatten())

    model.add(Dense(392))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.99, weights=None))#epsilon=1e-06, mode=0, momentum=0.9, weights=None
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    #model.add(Dense(784, activation="relu"))
    #model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="adam")

    return model 
'''
def train_and_evaluate_model():
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[weight_save_callback])

    score_fold = model.evaluate(X_total_test, Y_total_test, verbose=0)
    print('Fold test score: ', score_fold)
'''
#def __init__(self, n_splits=5, shuffle=True, random_state=None):
#    super(StratifiedKFold, self).__init__(n_splits, shuffle, random_state)

if __name__ == "__main__":
    fold = 0
    nsplits = 5
    X_total, Y_total, X_train, Y_train, X_test, Y_test = load_data()
    print("X_total shape from function:", X_total.shape)
    print("Y_total shape from function:", Y_total.shape)

    # skf = StratifiedKFold(n_splits=2)
    skf = StratifiedKFold(n_splits= nsplits, shuffle=True)
    skf.get_n_splits(X_total, Y_total)

    X_train1 = X_train.reshape(60000, 1, 784, 1)

    model = None
    model = create_model()
    model.fit(X_train1, Y_train, verbose=1, batch_size=128, nb_epoch=1)

    #score_first = model.evaluate(X_total_test, Y_total_test, verbose=0)
    #print('FIST test score: ', score_first)

    model.save('partly_trained.h5')
    del model

    print(skf)
    for train_index, test_index in skf.split(np.zeros(70000), np.zeros(70000)):
        fold += 1
        print ("Running fold #", fold)
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_total_train, X_total_test = X_total[train_index], X_total[test_index]
        Y_total_train, Y_total_test = Y_total[train_index], Y_total[test_index]

        X_total_train = X_total_train.reshape(X_total_train.shape[0], 1, 784, 1)
        X_total_test = X_total_test.reshape(X_total_test.shape[0], 1, 784, 1)
        '''
        print("X_total_train shape:", X_total_train.shape)
        print("X_total_train shape:", X_total_train.shape[0])
        print("X_total_test shape:", X_total_test.shape)
        print("Y_total_train shape:", Y_total_train.shape)
        print("Y_total_test shape:", Y_total_test.shape)
        '''
        #Y_total = np_utils.to_categorical(Y_total, nb_classes)

        model = load_model('partly_trained.h5')

        #Continue training

        model.fit(X_total_train, Y_total_train, verbose=1, batch_size=128, nb_epoch=2)
        score_fold = model.evaluate(X_total_test, Y_total_test, verbose=0)
        print('Fold test score: ', score_fold)
        
        model.save('partly_trained.h5')
        del model
        '''
        model = Sequential()

        model.add(Convolution2D(15, 3, 3, border_mode="same", activation="relu",input_shape=(1, 784, 1)))
        #model.add(Convolution2D(49, 3, 3, border_mode="same", activation="relu"))
        #model.add(Convolution2D(35, 3, 3, border_mode="same", activation="relu"))
        
        model.add(MaxPooling2D(pool_size=(1,2)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        
        model.add(Dense(32))
        model.add(BatchNormalization(epsilon=1e-06, mode=0, momentum=0.99, weights=None))#epsilon=1e-06, mode=0, momentum=0.9, weights=None
        #model.add(BatchNormalization())#epsilon=1e-06, mode=0, momentum=0.9, weights=None
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        #model.add(Dense(784, activation="relu"))
        #model.add(Dropout(0.2))

        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.compile(loss="categorical_crossentropy", optimizer="adam")

        #/home/mat/Downloads/
        checkpoint = ModelCheckpoint(filepath='/home/mat/Downloads/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=False, mode='auto')
        callbacks_list = [checkpoint]

        model.fit(X_total_train, Y_total_train, verbose=1, batch_size=128, nb_epoch=2, callbacks=callbacks_list, initial_epoch=0)
        
        #weight_save_callback = ModelCheckpoint('/path/to/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')
        #model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[weight_save_callback])

        score_fold = model.evaluate(X_total_test, Y_total_test, verbose=0)
        print('Fold test score: ', score_fold)
        #M = create_model()
        #print (M)
        #train_and_evaluate_model(M, X_total[train], Y_total[train], X_total[test], Y_total[test])
'''

model = load_model('partly_trained.h5')

X_train = X_train.reshape(60000, 1, 784, 1)
X_test = X_test.reshape(10000, 1, 784, 1)
X_total = X_total.reshape(70000, 1, 784, 1)

score_test = model.evaluate(X_test, Y_test, verbose=0)
print('Test score (the one that matters): ', score_test)

score_train = model.evaluate(X_train, Y_train, verbose=0)
print('Train score: ', score_train)

score_total = model.evaluate(X_total, Y_total, verbose=0)
print('Total score: ', score_total)

# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = model.predict_classes(X_test)

# Check which items we got right / wrong

#correct_indices = np.nonzero(predicted_classes == y_test)[0]
#incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

#predicted_classes_train = model.predict_classes(X_train)


#correct_indices_train = np.nonzero(predicted_classes_train == y_train)[0]
#incorrect_indices_train = np.nonzero(predicted_classes_train != y_train)[0]

'''
Using TensorFlow backend.
X_test original shape (10000, 28, 28)
y_test original shape (10000,)
X_train original shape (60000, 28, 28)
y_train original shape (60000,)
X_total shape (70000, 28, 28)
y_total shape (70000,)
Y_total shape (70000, 10)
X_total shape from function: (70000, 784)
Y_total shape from function: (70000, 10)
Epoch 1/1
60000/60000 [==============================] - 308s - loss: 0.1616
StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
Running fold # 1
Epoch 1/3
56000/56000 [==============================] - 231s - loss: 0.0811
Epoch 2/3
56000/56000 [==============================] - 231s - loss: 0.0566
Epoch 3/3
56000/56000 [==============================] - 234s - loss: 0.0430
Fold test score:  0.0551050563957
Running fold # 2
Epoch 1/3
56000/56000 [==============================] - 229s - loss: 0.0434
Epoch 2/3
56000/56000 [==============================] - 224s - loss: 0.0299
Epoch 3/3
56000/56000 [==============================] - 225s - loss: 0.0244
Fold test score:  0.0359240043175
Running fold # 3
Epoch 1/3
56000/56000 [==============================] - 221s - loss: 0.0284
Epoch 2/3
56000/56000 [==============================] - 221s - loss: 0.0199
Epoch 3/3
56000/56000 [==============================] - 221s - loss: 0.0151
Fold test score:  0.0162472057033
Running fold # 4
Epoch 1/3
56000/56000 [==============================] - 244s - loss: 0.0198
Epoch 2/3
56000/56000 [==============================] - 234s - loss: 0.0156
Epoch 3/3
56000/56000 [==============================] - 228s - loss: 0.0124
Fold test score:  0.00767371515938
Running fold # 5
Epoch 1/3
56000/56000 [==============================] - 255s - loss: 0.0132
Epoch 2/3
56000/56000 [==============================] - 229s - loss: 0.0116
Epoch 3/3
56000/56000 [==============================] - 234s - loss: 0.0085
Fold test score:  0.00586116705195
Test score (the one that matters):  0.00316790422798
Train score:  0.00381339114199
Total score:  0.0037211787257
10000/10000 [==============================] - 13s
Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x7f674c9515c0>>
Traceback (most recent call last):
'''