import json
import os

from src.master_thesis.preprocessing.augmentation import DataAugmenter
from src.master_thesis.data.loader.dataset_loader import DatasetLoader


class DatasetGenerator:

    def __init__(self, dataset_dir, augment=False, obscure_percent=0, batch_size=32):
        self.dataset_dir = dataset_dir
        self.dataset_names = os.listdir(dataset_dir)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(current_dir, '..', 'results', f'batch_size_{batch_size}')
        self.augment = augment
        self.batch_size = batch_size
        self.obscure_percent = obscure_percent
        self.data_augmentation = DataAugmenter() if augment else None

    def generate_combined_datasets(self):
        # TODO: Implement
        pass

    def generate_datasets(self, loader=DatasetLoader()):
        for dataset_name in self.dataset_names:
            dataset = loader.load_isic_data(os.path.join(self.dataset_dir, dataset_name), self.batch_size,
                                            obscure_images_percent=self.obscure_percent)
            if self.augment:
                dataset_name += "_augmented"
                dataset.train = self.data_augmentation.augment(dataset.train)
                dataset.val = self.data_augmentation.augment(dataset.val)
                dataset.test = self.data_augmentation.augment(dataset.test)

            yield dataset_name, dataset

    def save_results(self, model_name, dataset_name, results, epochs):
        results_filename = f'results_epochs_{epochs}.json'
        filepath = os.path.join(self.results_dir, results_filename)
        results_dir = os.path.dirname(filepath)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        if not os.path.exists(filepath):
            with open(filepath, 'w+') as file:
                json.dump({}, file)

        with open(filepath, 'r') as file:
            data = json.load(file)

        if model_name not in data:
            data[model_name] = {}

        data[model_name][dataset_name] = results

        with open(filepath, 'w+') as file:
            json.dump(data, file)
