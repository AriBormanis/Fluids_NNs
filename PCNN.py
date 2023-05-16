# -*- coding: utf-8 -*-
"""
Testing the PCNN on the turbulent data
"""

import tensorflow as tf
import keras.layers as layers
import numpy as np
import keras

from decoderHead import derivLayer
    
delta_x = 1/322
delta_y = 1/162
x_dim = 322
y_dim = 162

input_img = keras.Input(shape=(162, 322, 2))

x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# now we decode

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(100, activation='relu')(x)
x = layers.Reshape((10,10,1))(x)
# x = layers.Dense(16, activation='relu')(x)

decoded = derivLayer(delta_x, delta_y, padding='valid')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='MSE')
autoencoder.summary()

# train_data_in = np.load('train_data_in.npy')
# train_data_out = np.load('train_data_out.npy')

# train_data_in = train_data_in[:200,:,:,:]
# train_data_out = train_data_out[:200,:,:,:]

# autoencoder.fit(train_data_in, train_data_out, epochs=10, batch_size=32)

# autoencoder.save('first_autoencode_test')
