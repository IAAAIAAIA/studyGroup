
# coding: utf-8

# In[1]:

import keras
import numpy as np
from Matteos_Data_Loading_Methods import Data_Generators
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization


# In[2]:

model = Sequential()
model.add(Conv2D(64, (3,3), strides= (1,1), padding='same', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(128, (3, 3), padding= 'same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('softmax'))


# In[3]:

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])


# In[4]:

data = Data_Generators('/home/sexy/Documents/KaggleData/train-tif-v2',
                       '/home/sexy/Documents/KaggleData/train_v2.csv',
                       val_decimal=.02, train_decimal=.96, test_decimal=.02)

for batch_X, batch_y in zip(data.X_train(), data.y_train('primary')):
    batch_X = np.array(batch_X)
    batch_y = np.array(batch_y)
    model.fit(batch_X, batch_y, batch_size=8, epochs=1, verbose=1, 
              validation_data=(data.X_val(), data.y_val('primary')))
    


# In[5]:

score = model.evaluate(data.X_test(), data.y_test('primary'), verbose=0)
print(score)


# In[5]:

print('done')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



