# -*- coding: utf-8 -*-
"""
Function approximator

Created on Mon Mar 21 12:27:29 2022

@author: jstur2828
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dense

x = [i for i in range(-50, 51)]

y = [i**2.0 for i in x]

#plt.scatter(x,y)
#plt.title('input x vs output y')
#plt.xlabel('input variable x')
#plt.ylabel('output variable y')
#plt.show()

x = np.asarray([i for i in range(-50, 51)])

y = np.asarray([i**2.0 for i in x])

#print(x.min(), x.max(), y.min(), y.max())

x = x.reshape((len(x), 1))

y = y.reshape((len(x), 1))

scale_x = MinMaxScaler()
x = scale_x.fit_transform(x)

scale_y = MinMaxScaler()
y = scale_y.fit_transform(y)

#print(x.min(), x.max(), y.min(), y.max())

input_layer = keras.layers.Input(shape=(1,))
dense = Dense(10, activation='relu')(input_layer)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
output = Dense(1)(dense)

model = keras.Model(inputs=input_layer, outputs=output)

model.compile(loss='mse', optimizer='adam')

model.fit(x, y, epochs=500, batch_size=10, verbose=0)

yhat = model.predict(x)

x_plot = scale_x.inverse_transform(x)
y_plot = scale_y.inverse_transform(y)
yhat_plot = scale_y.inverse_transform(yhat)

print('MSE: %.3f' % mean_squared_error(y_plot, yhat_plot))

x1 = [i for i in range(-50, 51)]

y1 = [i**2.0 for i in x1]

plt.scatter(x_plot,yhat_plot, label='Predicted')
plt.scatter(x1,y1, label = 'actual')
#plt.scatter(x1,y1, label = 'actual')
plt.title('Input (x) versus Output (y)')
plt.xlabel('Input Variable (x)')
plt.ylabel('Output Variable (y)')
plt.legend()
plt.show()
