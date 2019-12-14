from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile, sys, os
sys.path.insert(0, os.path.abspath('.'))

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import tensorflow.compat.v1.keras.backend as K

# Import DeepExplain
from deepexplain.tf.v2_x import DeepExplain
mnist = tf.keras.datasets.mnist

# data parameter
batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = (x_train - 0.5) * 2
x_test = (x_test - 0.5) * 2

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]


y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
# note: labels do NOT need to be one hot encoded ??
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

# Build basic model
model = tf.keras.models.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                activation='relu',
                input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
# ^ IMPORTANT: notice that the final softmax must be in its own layer
# if we want to target pre-softmax units

model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
# run model
model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=2)
model.summary()
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# tf.compat.v1.keras.backend.get_session()
with DeepExplain(session=sess) as de:  # <-- init DeepExplain context
    # Need to reconstruct the graph in DeepExplain context, using the same weights.
    # With Keras this is very easy:
    # 1. Get the input tensor to the original model
    input_tensor = model.layers[0].input
    output_tensor = model.layers[-2].output

    print(input_tensor)
    print(output_tensor)
    # 2. We now target the output of the last dense layer (pre-softmax)
    # To do so, create a new model sharing the same layers untill the last dense (index -2)
    fModel = Model(inputs=input_tensor, outputs=output_tensor)

    target_tensor = fModel(input_tensor)

    xs = x_test[0:7]
    ys = y_test[0:7]
    print(xs.shape)
    print(ys.shape)

    attributions_gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)