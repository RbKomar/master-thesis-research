# src/data/augmentation.py
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

class DataAugmenter:

    def __init__(self, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True):
        self.data_augmentation = tf.keras.Sequential([
            preprocessing.Rescaling(1./255),
            preprocessing.RandomRotation(rotation_range),
            preprocessing.RandomTranslation(width_shift_range, height_shift_range),
            preprocessing.RandomZoom(zoom_range),
            preprocessing.RandomFlip(mode='horizontal' if horizontal_flip else None)
        ])

    def augment(self, dataset):
        return dataset.map(lambda x, y: (self.data_augmentation(x, training=True), y))
