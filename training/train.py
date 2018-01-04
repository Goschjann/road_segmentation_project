""""
Script to train model on any train data set and store it afterwards for the fcn_test.py script.

Input: set of train images as preprocessed by split_images.py

Output: trained neural net as .h5 file
"""


from __future__ import division, print_function, absolute_import

import os
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, merge, Dropout
from keras.initializers import RandomNormal
from keras.models import Model
from keras.callbacks import CSVLogger, EarlyStopping
from keras.regularizers import l2

import custommodule
custommodule.checker()

train_folder = storage = "/home/jgucci/Desktop/mnih_data/train/preprocessed/"

# Load label masks and satellite images
labels = custommodule.read_label_masks(path = train_folder,
                                       mask_name = 'lab',
                                       suffix = '.png',
                                       amount_masks = 400,
                                       crop_pixels = 0,
                                       output_shape = [512, 512],
                                       progress_step = 100,
                                       separation = False)

satellites = custommodule.read_input_maps(path = train_folder,
                                          image_name = 'sat',
                                          suffix = '.png',
                                          amount_maps = 400,
                                          crop_pixels = 0,
                                          output_shape = [512, 512, 3],
                                          progress_step = 100)

# Apply noise to the labels
# We found that noise addition does not increase performance for our architecture. Still we keep the function for
# experimental purposes
gaussian_labels = custommodule.gaussit(input_list = labels, sigma = 0)
gaussian_labels = gaussian_labels.reshape(-1, 512, 512, 1)

# Split in Train and validation data using sklearn's function
x_train, x_valid, y_train, y_valid = train_test_split(satellites, gaussian_labels, train_size = 0.9, random_state = 1337)

# Model architecture and weight initialization
init = RandomNormal(mean = 0, stddev = 0.01, seed = None)

input = Input(name = 'data_input', shape = (512, 512, 3,))

conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(drop5))
merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv6))
merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv7))
merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    UpSampling2D(size=(2, 2))(conv8))
merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

model = Model(input = input, output = conv10)

# print the model summary to the command line
model.summary()

# store temporary files
if not os.path.exists('/tmp/keras/'):
    os.makedirs('/tmp/keras/')

# Define optimizer for the model
adam = optimizers.Adam(lr = 0.0001,
                       beta_1 = 0.9,
                       beta_2 = 0.999,
                       epsilon = 1e-08,
                       decay = 1e-08)

# Define loss function and and metric
model.compile(optimizer = adam,
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# define callbacks: CSVLogger to save results and EarlyStopping to avoid overfitting
csv_logger = CSVLogger('mnih_040118_3.log')

stop_train = EarlyStopping(monitor = 'val_loss',
                           min_delta = 0,
                           patience = 5,
                           verbose = 0,
                           mode = 'auto')

# Execute model
model.fit(x_train, y_train,
          callbacks=[csv_logger, stop_train],
          validation_data=[x_valid, y_valid],
          batch_size = 1,
          epochs = 10,
          verbose = 2,
          shuffle = True)

# Save model to hard drive (roughly 400 mb)
print('done with training')
model.save('mnih_040118_3.h5')
print('done with saving')
