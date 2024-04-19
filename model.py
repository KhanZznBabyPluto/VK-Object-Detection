import numpy as np
import pandas as pd
from rmse import RMSE
import tensorflow as tf
from data_preprocessing import data_processing, image_processing

train_data_path = 'train.csv'

train_data, test_data = data_processing(train_data_path)

train_images = image_processing(train_data[:, 0]) / 255.0
test_images = image_processing(test_data[:, 0]) / 255.0

train_labels = train_data[:, 1].astype(int)
test_labels = test_data[:, 1].astype(int)

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae', RMSE()])

print(train_labels.shape)
print(train_images.shape)

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))