# -*- coding: utf-8 -*-
"""
Function approximator for problem 2

Created on Tue Mar 22 19:59:59 2022

@author: jstur2828
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, GaussianNoise
from mpl_toolkits import mplot3d

# Function to save raw model data (loss for each epoch)

def saveHistory():
    hist_df = pd.DataFrame(history.history) 
    hist_csv_file = 'history_function2.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

# Create two evenly spaced intervals for x and y

x = np.array(np.linspace(-5, 5, 100))
y = np.array(np.linspace(-5, 5, 100))

# Create a grid of x and y values (numpy meshgrid works well with linspace arrays)

X, Y = np.meshgrid(x, y)

# Create function that we will be predicting against

Z = X**4.0 - 22 * X**2.0 + Y**4.0 - 22 * Y ** 2.0

batch_size = 10

# Create model

input_layer = keras.layers.Input(shape=(200,))
dense = Dense(10, activation='relu')(input_layer)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
dense = Dense(10, activation='relu')(dense)
output = Dense(100)(dense)

# Build model

model = keras.Model(inputs=input_layer, outputs=output)

# Compile model

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# "Stack" the X and Y grids to produce a sequence that represents a
# Cartesian product that can be read by the computer

X1 = np.hstack((X, Y))

# Fit and run the model and save to history object for later use

history = model.fit(X1, Z, epochs=1000, batch_size=10, verbose=0)

saveHistory()

# Make predictions for Z

zhat = model.predict(X1)

print('MSE: %.3f' % mean_squared_error(Z, zhat))

# Plot the predictions on a 3D grid

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, zhat, rstride=1, cstride=1, cmap="viridis", linewidth=0, antialiased=False)

plt.show()

# Plot the loss on a graph

plt.plot(history.history['loss'])
plt.title('function 2 loss (no noise)')
plt.ylabel('loss')
plt.xlabel('number of epochs')
plt.show()