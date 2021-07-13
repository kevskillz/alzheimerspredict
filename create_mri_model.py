
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
from keras.saving.save import load_model
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.metrics import *
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization, GlobalAveragePooling2D



dimensions = (180, 180)


def plot_images(filepaths, grid_size):
    
    if grid_size > 6:
        grid_size = 5
    size = len(filepaths)
    plt.figure("Input Images", figsize=(grid_size + 5, grid_size + 5))
    
    for i in range(grid_size * grid_size):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.axis('off')
        img = filepaths[random.randint(0, size - 1)]
        plt.imshow(mpimg.imread(img), cmap='gray')
        
        plt.title(img.split('\\')[1])
    # plt.show()

# allows for multi-dimensional data

idg = ImageDataGenerator(width_shift_range=0.1, rotation_range=15,
                              height_shift_range=0.1, shear_range=0.1,
                              rescale=1/255,                         
                              zoom_range=0.1,
                              horizontal_flip=True,
                              fill_mode='constant')


# load training data
train = idg.flow_from_directory('./images', target_size=dimensions, shuffle=False, batch_size=6400)
print(train.class_indices)


# {'MildDemented': 0, 'ModerateDemented': 1, 'NonDemented': 2, 'VeryMildDemented': 3}

# plot_images(train.filepaths, 4)

data, l = train.next()


smote = SMOTE()

X, y = smote.fit_resample(data.reshape(-1, dimensions[0] * dimensions[1] * 3), l)

X = data.reshape(-1, dimensions[0], dimensions[1], 3)


train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size = 0.15)
train_data, val_data, train_labels, val_labels = train_test_split(X, y, test_size = 0.15)



incept = InceptionV3(False, "imagenet", input_shape=(180, 180, 3))

for i in incept.layers:
    i.trainable = False


model = Sequential()

model.add(incept)
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(BatchNormalization())

model.add(Dense(256, 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(128, 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(64, 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(32, 'relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(4, 'softmax')) # num of outputs
model.summary()





check = ModelCheckpoint(filepath='model-{epoch}.model', save_best_only=True, monitor='val_loss')

    
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                              metrics=['acc'])

model.summary()


EPOCHS = 50

history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=EPOCHS, callbacks=[check])
