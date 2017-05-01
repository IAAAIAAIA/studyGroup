import os
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Input
from keras.layers.merge import Average, Concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, Convolution3D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers.advanced_activations import PReLU
from keras import backend as K #enable tensorflow functions
from keras.layers.pooling import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from snapshot import SnapshotCallbackBuilder
import keras.metrics as metrics
import pandas as pd
import PIL
import csv
import time
import progressbar as pb

PATH = os.path.dirname(os.path.abspath(__file__))

def load_X_train_data(n_images):
    # Chose path to the folder containing the training data in .jpg format:
    train_data_path = '/home/sexy/CS231n/mat/Kaggle/train-jpg'
    # Chose number of images to load.
    # Type 'all' to load all the images in the folder, or 'half' to load half of them

    print('Loading Train Data: ')
    X_train, X_name_of_each_train = load_jpg_images(train_data_path, n_images)
    X_train = np.array(X_train)

    print('Shape or train images array: ', X_train.shape)
    return X_train, X_name_of_each_train

def load_X_test_data(n_images):
    # Chose path to the folder containing the test data in .jpg format:
    test_data_path = '/home/sexy/CS231n/mat/Kaggle/test-jpg'
    # Chose number of images to load.
    # Type 'all' to load all the images in the folder, or 'half' to load half of them
    print('Loading Test Data: ')
    X_test, X_name_of_each_test = load_jpg_images(test_data_path, n_images)
    X_test = np.array(X_test)

    print('Number of test images: ',  X_test.shape)
    return X_test, X_name_of_each_test

def load_Y_data(n_images):
    # Chose path to the .csv file containing the labels: 
    csv_path = '/home/sexy/CS231n/mat/Kaggle/train.csv'

    image_and_tags = csv_reader(csv_path)[:n_images]
    labels = label_lister(image_and_tags)
    Y_train = list_to_vec(image_and_tags['tags'], labels)
    return image_and_tags, labels, Y_train

def getkey(item):
    return item[0]

def load_jpg_images(folder, N):
    _list = os.listdir(folder)
    if N is 'all':
        N = int(len(_list))
    elif N is 'half':
        N = int(len(_list)/2)
    _list_n = [(int(''.join(list(filter(str.isdigit, x)))), _list[i]) for i, x in enumerate(_list)]
    # print(_list_n[0])
    _list_n = sorted(_list_n, key=getkey)
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), ], max_value=N).start() # max_value=len(list)).start()
    images = []
    filenames = []
    for i, _filename in enumerate(_list_n):
        if i >= N:
            break
        # print("\n", _filename[1], "testing:)")
        filename = _filename[1]
        img = np.array(PIL.Image.open(os.path.join(folder, filename)))/255
        if img is not None:
            images.append(img)
            filenames.append(filename)
        pbar.update(i)
    pbar.finish()
    return images, filenames

def csv_reader(file_labels):
    with open(file_labels) as f:
        CSVread = pd.read_csv(f)
    print('Labels succesfully loaded')
    return CSVread

def label_lister(labels_df):
    label_list = []
    for tag_str in labels_df.tags.values:
        labels = tag_str.split(' ')
        for label in labels:
            if label not in label_list:
                label_list.append(label)
    return label_list

def list_to_vec(list_img_labels, all_labels):
    number_of_labels = len(all_labels)
    number_of_pics = len(list_img_labels)
    vec = np.zeros([number_of_pics, number_of_labels], dtype=int)
    # ['haze', 'primary', 'agriculture', 'clear', 'water', 'habitation', 'road', 'cultivation', 'slash_burn', 'cloudy', 'partly_cloudy',
    # 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']
    print('Translating lables into vectors:')
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), ], max_value=(number_of_pics-1)).start()
    list_img_labels = [labels.split(' ') for labels in list_img_labels]
    for i in range(number_of_pics):
        pbar.update(i)
        for j in range(number_of_labels):
            if all_labels[j] in list_img_labels[i]:
                vec[i][j] = 1
            else:
                vec[i][j] = 0
    pbar.finish()
    return vec

# def vec_to_list(vec, all_labels)
#     number_of_labels = len(all_labels)
#     number_of_pics = len(list_img_labels)

# def vec_to_lis():

def train(run=0):
    X_train, X_name_of_each_train = load_X_train_data(5000)
    # X_test, X_name_of_each_test = load_X_test_data()
    image_and_tags, labels, Y_train = load_Y_data(X_train.shape[0])
    # print('number of tags :', np.array(labels).shape)
# train()
    #datagen = ImageDataGenerator(rotation_range=45,width_shift_range=0.2, height_shift_range=0.2)
    #datagen.fit(X_train)
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    nb_epoch = 80
    nb_musicians = 5
    snapshot = SnapshotCallbackBuilder(nb_epoch, nb_musicians, init_lr=0.01)
    #model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
    #                samples_per_epoch=len(X_train), nb_epoch=nb_epoch)
    model.fit(X_train, Y_train,
              batch_size=2,
              epochs=nb_epoch,
              verbose=1,
            #   validation_data=(X_test, Y_test),
              callbacks=snapshot.get_callbacks('snap-model'+str(run)))

    model.load_weights("weights/%s-Best.h5" % ('snap-model'+str(run)))
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['mean_squared_error'])
    score = model.evaluate(X_test, Y_test,
                           verbose=0)
    print('--------------------------------------')
    print('model'+str(run)+':')
    print('Test loss:', score[0])
    print('error:', str((1.-score[1])*100)+'%')
    return score

def create_model():
    _input = Input((256, 256, 4))
    incep1 = inception_net(_input)
    out = incep1
    model = Model(inputs=_input, outputs=[out])
    return model

def dropconnect_lambda():
    pass

def inception_net(_input):
    #x = Reshape((28, 28, 1))(_input)
    #x = Conv2D(32, (3, 3), strides=((1, 1)))(x)
    #x = Activation('relu')(x)
    x = Conv2D(16, (3, 3),strides=(2, 2))(_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(48, (3, 3),strides=((1, 1)))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(((3, 3)), strides=(2,2))(x)
    x = mniny_inception_module(x, 1)
    x = mniny_inception_module(x, 2)
    #x = MaxPooling2D(((3, 3)), strides=(2,2))(x)
    x = mniny_inception_module(x, 2)
    x, soft1 = mniny_inception_module(x, 3, True)
    x = mniny_inception_module(x, 3)
    x = mniny_inception_module(x, 3)
    x, soft2 = mniny_inception_module(x, 4, True)
    x = MaxPooling2D(((3, 3)), strides=(2,2))(x)
    x = mniny_inception_module(x, 4)
    x = mniny_inception_module(x, 5)
    x = AveragePooling2D(((5, 5)), strides=((1, 1)))(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(17)(x)
    soft3 = Activation('hard_sigmoid')(x)
    out = Average()([soft1, soft2, soft3])
    return out

def mniny_inception_module(x, scale=1, predict=False):

    ###x is input layer, scale is factor to scale kernel sizes by

    x11 = Conv2D(int(16*scale), (1, 1), padding='valid')(x)
    x11 = BatchNormalization()(x11)
    x11 = Activation('relu')(x11)

    x33 = Conv2D(int(24*scale), (1, 1))(x)
    x33 = BatchNormalization()(x33)
    x33 = Activation('relu')(x33)
    x33 = Conv2D(int(32*scale), (3, 3), padding='same')(x33)
    x33 = BatchNormalization()(x33)
    x33 = Activation('relu')(x33)

    x55 = Conv2D(int(4*scale), (1, 1))(x)
    x55 = BatchNormalization()(x55)
    x55 = Activation('relu')(x55)
    x55 = Conv2D(int(8*scale), (5, 5), padding='same')(x55)
    x55 = BatchNormalization()(x55)
    x55 = Activation('relu')(x55)

    x33p = MaxPooling2D(((3, 3)), strides=((1, 1)), padding='same')(x)
    x33p = Conv2D(int(8*scale), (1, 1))(x33p)
    x33p = BatchNormalization()(x33p)
    x33p = Activation('relu')(x33p)

    out = Concatenate(axis=3)([x11, x33, x55, x33p])

    if predict:
        predict = AveragePooling2D(((5, 5)), strides=((1, 1)))(x)
        predict = Conv2D(int(8*scale), (1, 1))(predict)
        predict = BatchNormalization()(predict)
        predict = Activation('relu')(predict)
        predict = Dropout(0.3)(predict)
        predict = Flatten()(predict)
        predict = Dense(120)(predict)
        predict = BatchNormalization()(predict)
        predict = Activation('relu')(predict)
        predict = Dense(17)(predict)
        predict = Activation('hard_sigmoid')(predict)
        return out, predict

    return out

def test_model():
    model = create_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print("MODEL COMPILES SUCCESSFULLY")

def evaluate_ensemble(Best=True):
    
    ###loads and evaluates an ensemle from the models in the model folder.
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(10000, 784)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = np_utils.to_categorical(y_test, 10)

    model_dirs = []
    for i in os.listdir('weights'):
        if '.h5' in i:
            if not Best:
                model_dirs.append(i)
            else:
                if 'Best' in i:
                    model_dirs.append(i)

    preds = []
    model = create_model()
    for mfile in model_dirs:
        print(os.path.join('weights',mfile))
        model.load_weights(os.path.join('weights',mfile))
        yPreds = model.predict(X_test, batch_size=128, verbose=0)
        preds.append(yPreds)

    weighted_predictions = np.zeros((X_test.shape[0], 10), dtype='float64')
    weight = 1./len(preds)
    for prediction in preds:
        weighted_predictions += weight * prediction
    y_pred =weighted_predictions

    print(type(Y_test))
    print(type(y_pred))
    Y_test = tf.convert_to_tensor(Y_test)
    y_pred = tf.convert_to_tensor(y_pred)
    print(type(Y_test))
    print(type(y_pred))

    loss = metrics.categorical_crossentropy(Y_test, y_pred)
    acc = metrics.categorical_accuracy(Y_test, y_pred)
    sess = tf.Session()
    print('--------------------------------------')
    print('ensemble')
    print('Test loss:', loss.eval(session=sess))
    print('error:', str((1.-acc.eval(session=sess))*100)+'%')
    print('--------------------------------------')

def evaluate(eval_all=False):

    ###evaluate models in the weights directory,
    ###defaults to only models with 'best'

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(10000, 784)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = np_utils.to_categorical(y_test, 10)
    evaluations = []


    for i in os.listdir('weights'):
        if '.h5' in i:
            if eval_all:
                evaluations.append(i)
            else:
                if 'Best' in i:
                    evaluations.append(i)
    print(evaluations)
    model = create_model()
    for run, i in enumerate(evaluations):
        model.load_weights(os.path.join('weights',i))
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                    metrics=['categorical_accuracy'])
        score = model.evaluate(X_test, Y_test,
                            verbose=0)
        print('--------------------------------------')
        print('model'+str(run)+':')
        print('Test loss:', score[0])
        print('error:', str((1.-score[1])*100)+'%')

#evaluate_ensemble(True)
#evaluate(eval_all=False)
#test_model()


# To train:
run = 0
while True:
   train(run)
   run += 1




# Performance history (notable cases):
#Ensemble (2 fire_net, 3 conv_net):
#Epoch 30/30
#60000/60000 [==============================] - 33s - loss: 0.0053 - val_loss: 0.0229
#10000/10000 [==============================] - 8s
#Test score: 0.0229417834882
