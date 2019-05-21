import tensorflow as tf
import keras
import numpy as np

#class mycallbacks (tf.keras.callbacks.Callback):
#    def iflossless_before_epochend(self, epoch, log={}):
#        if(log.get(epoch) < .3):
#            print("loss less than 30%, so ending the epochs")
#            model.stop_training = True

#callback1 = mycallbacks()
dataset = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

train_images = train_images.reshape(60000,28,28,1)
train_images = train_images / 255

test_images = test_images.reshape(10000,28,28,1)
test_images = test_images / 255

model = keras.Sequential([
    keras.layers.Conv2D(64, (3,3),activation= 'relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer= tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy')
model.summary
model.fit(train_images, train_labels, epochs=15)
