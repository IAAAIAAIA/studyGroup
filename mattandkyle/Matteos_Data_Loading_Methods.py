
# coding: utf-8

# In[5]:

import keras
import csv
import os


# In[1]:

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


# In[ ]:



