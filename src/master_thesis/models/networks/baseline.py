from keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

from src.master_thesis.models.train_model import ModelHandler


class BaselineModel:
    """Class responsible for creating the baseline model."""

    @staticmethod
    def create(num_classes, input_shape=(150, 150, 3)):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(num_classes, activation='sigmoid')  # Assuming binary classification
        ])
        return model


class BaselineModelHandler(ModelHandler):

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

    def initialize_model(self):
        weights = 'imagenet' if self.use_imagenet else None
        base_model = BaselineModel(weights=weights, include_top=False, input_shape=self.input_shape)

        # Add custom layers
        x = base_model.output
        x = self.custom_layers(x)
        predictions = Dense(self.num_classes, activation='sigmoid' if self.num_classes == 1 else 'softmax')(x)

        # This is the model we will train
        self.model = Model(inputs=base_model.input, outputs=predictions)
