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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dense
from scipy.interpolate import make_interp_spline, BSpline
from mpl_toolkits import mplot3d

x = np.array(np.linspace(-5, 5, 100))
y = np.array(np.linspace(-5, 5, 100))

X, Y = np.meshgrid(x, y)

Z = X**4.0 - 22 * X**2.0 + Y**4.0 - 22 * Y ** 2.0

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

input_layer = keras.layers.Input(shape=(X.shape[1],))
dense = Dense(10, activation='relu')(input_layer)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
output = Dense(2)(dense)

model = keras.Model(inputs=input_layer, outputs=output)

model.compile(loss='mse', optimizer='adam')

model.fit(X, Z, epochs=1000, batch_size=10, verbose=0)

zhat = model.predict(X)

print('MSE: %.3f' % mean_squared_error(Z, zhat))