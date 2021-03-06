import logging
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD
import numpy as np

class Model(object):
    def __init__(self):
        input_length = 18
        hidden_length = 20
        self._model = Sequential([
            # Dropout(0.2, input_shape=(18,)),
            # Dense(hidden_length),
            Dense(hidden_length, input_dim = input_length),
            # Activation('relu'),
            Activation('sigmoid'),
            BatchNormalization(),
            # Dense(10),
            # Activation('relu'),
            # BatchNormalization(),
            Dense(6),
            Activation('softmax')
            # Dense(6, input_dim = input_length),
            # Activation('softmax')
            ])
        # optimizer = SGD()
        optimizer = RMSprop()
        self._model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self._model.predict(np.array([[0]*input_length])) # dummy computation to warm up model
    def predict(self, input):
        return self._model.predict(input)
