import logging
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

class Model(object):
    def __init__(self):
        input_length = 18
        hidden_length = 20
        self._model = Sequential([
            Dense(hidden_length, input_dim = input_length),
            Activation('relu'),
            Dense(10),
            Activation('relu'),
            Dense(10),
            # Activation('relu'),
            # Dense(10),
            # Activation('relu'),
            # Dense(10),
            # Activation('relu'),
            Dense(6),
            Activation('softmax')
            # Dense(6, input_dim = input_length),
            # Activation('softmax')
            ])
        self._model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # self._model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')
        self._model.predict(np.array([[0]*input_length])) # dummy computation to warm up model
    def predict(self, input):
        return self._model.predict(input)
