from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from models.train_model import ModelTrainer


def create_model(num_classes, input_shape=(150, 150, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='sigmoid')  # zakładając, że mamy problem binarnej klasyfikacji
    ])
    return model


class BaselineTrainer(ModelTrainer):
    def __init__(self, num_classes=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.model = None

    def initialize_model(self):
        self.model = create_model(self.num_classes, input_shape=self.input_shape)
