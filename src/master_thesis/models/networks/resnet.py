from tensorflow.keras.applications import ResNet152
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model

from src.master_thesis.models.train_model import ModelHandler


class ResNetModelHandler(ModelHandler):
    def __init__(self, num_classes=1, use_imagenet=True, custom_layers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.use_imagenet = use_imagenet
        self.custom_layers = custom_layers if custom_layers else self.default_custom_layers
        self.model = None

    @staticmethod
    def default_custom_layers(x):
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        return x

    def _initialize_model(self):
        weights = 'imagenet' if self.use_imagenet else None
        base_model = ResNet152(weights=weights, include_top=False, input_shape=self.input_shape)

        # Add custom layers
        x = base_model.output
        x = self.custom_layers(x)
        predictions = Dense(self.num_classes, activation='sigmoid' if self.num_classes == 1 else 'softmax')(x)

        # This is the model we will train
        self.model = Model(inputs=base_model.input, outputs=predictions)
