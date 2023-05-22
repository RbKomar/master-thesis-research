# src/models/model_builder.py

from tensorflow.keras.applications import VGG16, ResNet50
from src.models.baseline import BaselineModel

class ModelBuilder:

    @staticmethod
    def get_model(model_name: str):
        if model_name == 'baseline':
            return BaselineModel()
        elif model_name == 'VGG16':
            return VGG16(weights=None, classes=7)  # Assume 7 classes for skin lesions
        elif model_name == 'ResNet':
            return ResNet50(weights=None, classes=7)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
