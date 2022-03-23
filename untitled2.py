# -*- coding: utf-8 -*-
"""
Function approximator for problem 2

Created on Tue Mar 22 19:59:59 2022

@author: jstur2828
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, GaussianNoise
from mpl_toolkits import mplot3d

x = np.array(np.linspace(-5, 5, 100))
y = np.array(np.linspace(-5, 5, 100))

X, Y = np.meshgrid(x, y)

Z = X**4.0 - 22 * X**2.0 + Y**4.0 - 22 * Y ** 2.0

batch_size = 10

input_layer = keras.layers.Input(batch_size = batch_size, shape=(200,))
dense = Dense(10, activation='relu')(input_layer)
#gauss = GaussianNoise(stddev=50)(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
output = Dense(100)(dense)

model = keras.Model(inputs=input_layer, outputs=output)

model.compile(loss='mse', optimizer='adam')

X1 = np.hstack((X, Y))

model.fit(X1, Z, epochs=1000, batch_size=10, verbose=0)

zhat = model.predict(X1)

print('MSE: %.3f' % mean_squared_error(Z, zhat))

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, zhat, rstride=1, cstride=1, cmap="viridis", linewidth=0, antialiased=False)

plt.show()

'''
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
x = np.array(np.linspace(-5, 5, 100))
y = np.array(np.linspace(-5, 5, 100))

X, Y = np.meshgrid(x, y)
Z = X**4.0 - 22 * X**2.0 + Y**4.0 - 22 * Y ** 2.0

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="viridis", linewidth=0, antialiased=False)

plt.show()
'''
