# -*- coding: utf-8 -*-
"""
Function approximator for problem 1

Created on Tue Mar 22 18:33:19 2022

@author: jstur2828
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, GaussianNoise
from scipy.interpolate import make_interp_spline, BSpline

# Function to save raw model data (loss for each epoch)

def saveHistory():
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = 'history_function1.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

# Create a range of x values from -5 to 5

x = [i for i in range(-5, 5)]
x1 = np.asarray(x)

# Create a new range of values in order to plot a smooth quadratic line
xnew = np.linspace(-5.5, 5.5, 300)

# Calculate the predicted values in y

y = [(i**4.0 - 22 * (i ** 2.0)) for i in x]

# Make a spline in order to smoothly fit a quadratic function

xnew = np.linspace(-5.5, 5.5, 300)
spl = make_interp_spline(x, y, k=3)
power_smooth = spl(xnew)

y = [(i**4.0 - 22 * (i ** 2.0)) for i in xnew]
y1 = np.asarray(y)

# Create model 

input_layer = keras.layers.Input(shape=(1,))
dense = Dense(10, activation='relu')(input_layer)
gauss = GaussianNoise(stddev=20)(dense)
dense = Dense(10, activation='relu')(gauss)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
output = Dense(1)(dense)

# Build model

model = keras.Model(inputs=input_layer, outputs=output)

# Compile model 

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

# Fit and run the model and save to history object for later use

history = model.fit(xnew, y1, epochs=10000, batch_size=10, verbose=0)

saveHistory()

# Make predictions for y

yhat = model.predict(xnew)

print('MSE: %.3f' % mean_squared_error(y1, yhat))

# Plot the predicted vs. actual values on a 2D graph

spl1 = make_interp_spline(xnew, yhat, k=3)
power_smooth1 = spl1(xnew)

plt.plot(xnew,power_smooth1, label='Predicted')
plt.plot(xnew,power_smooth, label = 'Actual')
plt.title('Input (x) versus Output (y)')
plt.xlabel('Input Variable (x)')
plt.ylabel('Output Variable (y)')
plt.legend()
plt.show()

# Plot the loss on a graph

plt.plot(history.history['loss'])
plt.title('function 1 loss (without noise)')
plt.ylabel('loss')
plt.xlabel('number of epochs')
plt.show()
