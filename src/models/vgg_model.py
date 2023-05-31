from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from models.train_model import ModelTrainer


class VGGModelTrainer(ModelTrainer):

    def __init__(self, num_classes=1, use_imagenet=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.use_imagenet = use_imagenet
        self.model=None

    def initialize_model(self):
        if self.use_imagenet:
            weights = 'imagenet'
        else:
            weights = None
        base_model = VGG16(weights=weights, include_top=False, input_shape=self.input_shape)

        # Add custom layers
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(self.num_classes, activation='sigmoid')(x)

        # This is the model we will train
        self.model = Model(inputs=base_model.input, outputs=predictions)