# src/models/mobilenet_model.py
from tensorflow.keras.applications import MobileNet
from src.models.model_trainer import ModelTrainer

class MobileNetModelTrainer(ModelTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = MobileNet(weights='imagenet', include_top=False, input_shape=self.input_shape)
