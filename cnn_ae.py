# Test 12
from tensorflow.keras.layers import Input, Conv1D, Dropout, Conv1DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.models import load_model, save_model

import tensorflow as tf
import os
import random
import numpy as np

from tensorflow.keras.layers import BatchNormalization

# Test 12
class Conv_AE_12:     
    def __init__(self):
        self._Random(0)
        
    def _Random(self, seed_value):      
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        
    def _build_model(self):
        
        model = Sequential(
            [
                Input(shape=(self.shape[1], self.shape[2])),
                Conv1D(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                BatchNormalization(),
                # Dropout(rate=0.2),
                Conv1D(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Dropout(rate=0.2),
                Conv1D(
                    filters=8, kernel_size=7, padding="same", strides=1, activation="relu"
                ),
                Conv1DTranspose(
                    filters=8, kernel_size=7, padding="same", strides=1, activation="relu"
                ),
                Dropout(rate=0.2),
                Conv1DTranspose(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Conv1DTranspose(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        
        return model
    
    def fit(self, data, validation_split=0.1, epochs=40, verbose=0, shuffle=True, batch_size = 32):      
        self.shape = data.shape
        self.model = self._build_model()
        
        history = History()
        
        return self.model.fit(
            data,
            data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0),history
            ],
        )

    def predict(self, data):       
        return self.model.predict(data)