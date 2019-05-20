#basic keras neural network with a single layer and single neuron
# if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.
# How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.

import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([1.0,2.0,3.0,4.0,5.0,6.0], dtype=float)
ys = np.array([1,1.5,2.0,2.5,3.0,3.5], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([7.0]))
