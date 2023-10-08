# src/models/resnet_model.py
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from models.train_model import ModelTrainer


class XceptionModelTrainer(ModelTrainer):
    def __init__(self, num_classes=1, use_imagenet=True, layers_to_train=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.use_imagenet = use_imagenet
        self.model = None
        self.layers_to_train = layers_to_train
    def initialize_model(self):
        weights = 'imagenet' if self.use_imagenet else None
        base_model = Xception(weights=weights, include_top=False, input_shape=self.input_shape)

        for layer in base_model.layers[:-self.layers_to_train]:
            layer.trainable = False

        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='sigmoid')(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)
