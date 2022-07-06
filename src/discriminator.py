import tensorflow as tf
import numpy as np

# simple CNN discriminator
from sacred import Experiment
from torch import nn

# singleton pattern applied for Discriminator
class Discriminator(nn.Module):

    instance = None

    @classmethod
    def get_instance(cls, input_len=64):
        if not cls.instance:
            return cls(input_len)
        return cls.instance

    def __init__(self, input_len=64):
        super(Discriminator, self).__init__()
        model = tf.keras.Sequential(name='DIS')
        img_shape = (512, 512, 3)

        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same',
                                         activation='relu', input_shape='img_shape'))

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same',
                                         activation='relu', input_shape='img_shape'))

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same',
                                         activation='relu', input_shape='img_shape'))

        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

