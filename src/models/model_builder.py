# src/models/model_builder.py
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from src.models.baseline import BaselineModel


class ModelBuilder:

    @staticmethod
    def get_model(model_name: str, input_shape: tuple, num_classes: int):
        if model_name == 'baseline':
            return BaselineModel(input_shape=input_shape, num_classes=num_classes)
        elif model_name == 'VGG16':
            base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
            return ModelBuilder.add_top_layers(base_model, num_classes)
        elif model_name == 'ResNet':
            base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)
            return ModelBuilder.add_top_layers(base_model, num_classes)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @staticmethod
    def add_top_layers(base_model, num_classes):
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        return model
