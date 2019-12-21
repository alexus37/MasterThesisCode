from utils import plot, plt, compare_images
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from datetime import datetime
import tensorflow as tf
import numpy as np

import tempfile
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

print(f'tf version = {tf.__version__}')

# consts
DATA_ROOT = './data/leapGestRecog'
IMAGE_SHAPE = (120, 320)
EPOCHS = 5
BATCH_SIZE = 64
NUM_CLASSES = 10
LOG_DIR = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_PATH = 'models/leapHandGesture.h5'

image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.1)
image_data_train = image_generator.flow_from_directory(
    str(DATA_ROOT),
    target_size=IMAGE_SHAPE,
    color_mode="grayscale",
    subset='training'
)
image_data_validation = image_generator.flow_from_directory(
    str(DATA_ROOT),
    target_size=IMAGE_SHAPE,
    color_mode="grayscale",
    subset='validation')

image_data_train.batch_size = BATCH_SIZE


label_names = ['Palm', 'L', 'Fist', 'Fist moved',
               'Thumb', 'Index', 'Ok', 'Palm moved', 'C', 'down']

model = tf.keras.models.Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(*IMAGE_SHAPE, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=['acc'])


history = model.fit_generator(image_data_train,
                    epochs=EPOCHS,
                    steps_per_epoch = image_data_train.samples // BATCH_SIZE,
                    validation_data = image_data_validation,
                    validation_steps = image_data_validation.samples // BATCH_SIZE)
# save the weights
model.save(MODEL_PATH)