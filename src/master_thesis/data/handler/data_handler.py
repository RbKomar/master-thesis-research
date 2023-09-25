from src.master_thesis.data.generator.dataset_generator import DataSetCreator
from src.master_thesis.data.loader.csv_loader import CSVLoader
from src.master_thesis.data.loader.dataset_loader import ImagePathProcessor
from src.master_thesis.data.management.dataset_manager import DatasetManager


class DataHandler:
    def __init__(self, data_path, img_ext, batch_size, image_parser, obscure_images_percent, train_prop, val_prop,
                 year) -> DatasetManager:
        csv_loader = CSVLoader(data_path, year)
        self.img_path_processor = ImagePathProcessor(csv_loader.load(), img_ext)
        img_paths, img_labels = self.img_path_processor.process_paths(data_path, train_prop, val_prop)
        self.dataset_creator = DataSetCreator(img_paths, img_labels, image_parser, obscure_images_percent, batch_size)
        self.datasets = self.dataset_creator.create_datasets()

    def get_datasets(self):
        dataset_manager = DatasetManager(self.datasets["train"], self.datasets["val"], self.datasets["test"])
        return dataset_manager
