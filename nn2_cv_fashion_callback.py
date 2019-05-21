import tensorflow as tf

class mycallbacks(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')>0.3):
            print("loss less than 30%, so ending the epochs")
            self.model.stop_train = True

import keras
import numpy as np
import matplotlib.pyplot as plt 

callbacks = mycallbacks()
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

train_images = train_images/255
test_images = test_images/255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy')
model.fit(train_images, train_labels, epochs=5, callbacks = [callbacks])

model.evaluate(test_images, test_labels)
