# -*- coding: utf-8 -*-
"""
Function approximator for problem 1

Created on Tue Mar 22 18:33:19 2022

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



x = [i for i in range(-5, 5)]
x1 = np.asarray(x)

xnew = np.linspace(-5.5, 5.5, 300)

y = [(i**4.0 - 22 * (i ** 2.0)) for i in x]

xnew = np.linspace(-5.5, 5.5, 300)
spl = make_interp_spline(x, y, k=3)
power_smooth = spl(xnew)

y1 = np.asarray(y)


input_layer = keras.layers.Input(shape=(1,))
dense = Dense(10, activation='relu')(input_layer)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
output = Dense(1)(dense)

model = keras.Model(inputs=input_layer, outputs=output)

model.compile(loss='mse', optimizer='adam')

model.fit(x1, y1, epochs=1000, batch_size=10, verbose=0)

yhat = model.predict(x1)

print('MSE: %.3f' % mean_squared_error(y1, yhat))

spl1 = make_interp_spline(x1, yhat, k=3)
power_smooth1 = spl1(xnew)

plt.plot(xnew,power_smooth1, label='Predicted')
plt.plot(xnew,power_smooth, label = 'Actual')
plt.title('Input (x) versus Output (y)')
plt.xlabel('Input Variable (x)')
plt.ylabel('Output Variable (y)')
plt.legend()
plt.show()
