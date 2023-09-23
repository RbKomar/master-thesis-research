# src/config.py
DATA_DIR = r'C:\Users\ml\PycharmProjects\master-thesis-research\data'
MODELS_DIR = r'C:\Users\ml\PycharmProjects\master-thesis-research\models'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


class Config:
    """Class to manage configuration settings."""

    def __init__(self):
        self.dataset_name = "HAM10000"
        self.file_path = 'results_epochs_10.json'
        self.epochs = EPOCHS
        self.show_in_web = True
        self.data_dir = DATA_DIR
        self.models_dir = MODELS_DIR
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE
