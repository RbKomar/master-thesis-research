import os

from src.master_thesis.data.handler.hashing import ImageHasher
from src.master_thesis.data.handler.augmentation import DataAugmenter
from src.master_thesis.data.loader.base_loader import BaseLoader
from src.master_thesis.data.management.results_manager import ResultsManager


class DatasetGenerator:

    def __init__(self, dataset_dir, augment=False, obscure_percent=0, batch_size=32):
        self.dataset_dir = dataset_dir
        self.dataset_names = os.listdir(dataset_dir)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(current_dir, '..', 'results', f'batch_size_{batch_size}')
        self.batch_size = batch_size
        self.obscure_percent = obscure_percent
        self.augmenter = DataAugmenter() if augment else None

    def generate_combined_datasets(self, loader=BaseLoader()):
            combined_dataset = []
            image_hasher = ImageHasher()

            for dataset_name in os.listdir(self.dataset_dir):

                dataset_path = os.path.join(self.dataset_dir, dataset_name)
                if not os.path.isdir(dataset_path):
                    continue  # Skip non-directory files

                dataset = loader.load(dataset_path)

                if self.augmenter:
                    dataset = self.augmenter.augment(dataset)

                for image, label in dataset:
                    unique_image = image_hasher.get_image_if_is_not_duplicate(image)

                    if unique_image is not None:
                        combined_dataset.append(
                            (unique_image, label))

            return combined_dataset

    def generate_datasets(self, loader=BaseLoader()):
        for dataset_name in self.dataset_names:
            dataset = loader.load(os.path.join(self.dataset_dir, dataset_name), self.batch_size,
                                            obscure_images_percent=self.obscure_percent)
            if self.augmenter:
                dataset_name += "_augmented"
                dataset.train = self.augmenter.augment(dataset.train)

            yield dataset_name, dataset

    def save_results(self, model_name, dataset_name, results, epochs):
        results_manager = ResultsManager(self.results_dir)
        results_manager.save_results(model_name, dataset_name, results, epochs)


