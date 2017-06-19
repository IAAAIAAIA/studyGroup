
# coding: utf-8

# In[8]:

import keras
import csv
import os
import cv2
import progressbar as pb
import numpy as np
import PIL
<<<<<<< HEAD
import pandas as pd
=======
>>>>>>> 7279a094a8311a26d04b1311639f10b6a59740f5


# In[10]:

def load_X_train_data(n_images):
    # Chose path to the folder containing the training data in .jpg format:
    train_data_path = '/home/sexy/Documents/KaggleData/train-tif-v2'
    # Chose number of images to load.
    # Type 'all' to load all the images in the folder, or 'half' to load half of them

    print('Loading Train Data: ')
    X_train, X_name_of_each_train = load_jpg_images(train_data_path, n_images)
    X_train = np.array(X_train)

    print('Shape or train images array: ', X_train.shape)
    return X_train, X_name_of_each_train

def load_X_test_data(n_images):
    # Chose path to the folder containing the test data in .jpg format:
    test_data_path = '/home/sexy/Documents/KaggleData/train-tif-v2'
    # Chose number of images to load.
    # Type 'all' to load all the images in the folder, or 'half' to load half of them
    print('Loading Test Data: ')
    X_test, X_name_of_each_test = load_jpg_images(test_data_path, n_images)
    X_test = np.array(X_test)

    print('Number of test images: ',  X_test.shape)
    return X_test, X_name_of_each_test

def load_Y_data(n_images):
    # Chose path to the .csv file containing the labels: 
    csv_path = '/home/sexy/Documents/KaggleData/train_v2.csv'

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
        img = np.array(cv2.imread(os.path.join(folder, filename)))/255
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
                vec[i][j] = -1
    pbar.finish()
    return vec

class Data_Generators:
    
    def __init__(self, X_path, y_path, train_decimal=.7, val_decimal=.2, test_decimal=.1, batch_size=500):
        #X_path should be path to folder containing image files
        #y_path should be a csv file of the labels
        #_decimal inputs should be the decimal portion of data that is of each set
        #all should add up to 1
        if round(train_decimal + val_decimal + test_decimal) != 1.0:
            raise ValueError('train_decimal, val_decimal, and test_decimal arguements should add up to 1')
        self.X_path = X_path
        self.y_path = y_path
        self.batch_size = batch_size
        total = len(os.listdir(self.X_path))
        self.num_train = (int)(total * train_decimal)
        self.num_val = (int)(total * val_decimal)
        self.num_test = total - (self.num_train + self.num_val)
        
    def csv_labels_to_numbers(self, array, labels, category):
        #used to convert array of all label int values to array of single binary value for each picture
        array = np.array(array)
        new_array = []
        if category in labels:
            cat_index = labels.index(category)
        else:
            raise ValueError('Incorrect category value')
        for i in range(array.shape[0]):
            new_array.append(array[i,cat_index])
        return new_array
    
    def X_train(self):
        file_list = os.listdir(self.X_path)
        _list_n = [(int(''.join(list(filter(str.isdigit, x)))), file_list[i]) for i, x in enumerate(file_list)]
        # print(_list_n[0])
        _list_n = sorted(_list_n, key=getkey)
        #pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), ], max_value=N).start() # max_value=len(list)).start()
        iterations = (int)(self.num_train / self.batch_size if self.num_train % self.batch_size is 0
                           else self.num_train / self.batch_size + 1)
        for i in range(iterations - 1):
            images = []
            if i*self.batch_size+self.batch_size < self.num_train:
                for _filename in _list_n[i*self.batch_size: i*self.batch_size+self.batch_size]:                                                                                                      
                    filename = _filename[1]
                    img = np.array(cv2.imread(os.path.join(self.X_path, filename)))/255
                    if img is not None:
                        images.append(img)
            else:
                for _filename in _list_n[i*self.batch_size: num_train]:                                                                                        
                    filename = _filename[1]
                    img = np.array(PIL.Image.open(os.path.join(self.X_path, filename)))/255
                    if img is not None:
                        images.append(img)
            yield images

    def y_train(self, category):
        iterations = (int)(self.num_train / self.batch_size if self.num_train % self.batch_size is 0 
                           else self.num_train / self.batch_size + 1)
        for i in range(iterations - 1):
            if self.batch_size*i+self.batch_size >= self.num_train:
                image_and_tags = csv_reader(self.y_path)[self.batch_size*i: num_train]
            else:
                image_and_tags = csv_reader(self.y_path)[self.batch_size*i: self.batch_size*i+self.batch_size]
            
            labels = label_lister(image_and_tags)
            Y_train_all = list_to_vec(image_and_tags['tags'], labels)
            Y_train = self.csv_labels_to_numbers(Y_train_all, labels, category)
            yield Y_train
            
    def X_val(self):
        file_list = os.listdir(self.X_path)
        _list_n = [(int(''.join(list(filter(str.isdigit, x)))), file_list[i]) for i, x in enumerate(file_list)]
        _list_n = sorted(_list_n, key=getkey)
        images = []
        for _filename in _list_n[self.num_train: self.num_train+self.num_val]:
            filename = _filename[1]
            img = np.array(cv2.imread(os.path.join(self.X_path, filename)))/255
            if img is not None:
                 images.append(img)
        images = np.array(images)
        return images
            
    def y_val(self, category):
        image_and_tags = csv_reader(self.y_path)[self.num_train: self.num_train+self.num_val]
        labels = label_lister(image_and_tags)
        Y_train_all = list_to_vec(image_and_tags['tags'], labels)
        Y_train = self.csv_labels_to_numbers(Y_train_all, labels, category)
        Y_train = np.array(Y_train)
        return Y_train
            

    def X_test(self):
        file_list = os.listdir(self.X_path)
        _list_n = [(int(''.join(list(filter(str.isdigit, x)))), file_list[i]) for i, x in enumerate(file_list)]
        _list_n = sorted(_list_n, key=getkey)
        images = []
        for _filename in _list_n[self.num_train+self.num_val:]:
            filename = _filename[1]
            img = np.array(cv2.imread(os.path.join(self.X_path, filename)))/255
            if img is not None:
                 images.append(img)
        images = np.array(images)
        return images
            
    def y_test(self, category):
        image_and_tags = csv_reader(self.y_path)[self.num_train+self.num_val:]
        labels = label_lister(image_and_tags)
        Y_train_all = list_to_vec(image_and_tags['tags'], labels)
        Y_train = self.csv_labels_to_numbers(Y_train_all, labels, category)
        images = np.array(images)
        return Y_train
            
   

# In[ ]:




# In[ ]:



