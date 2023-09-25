import json
import os

import tensorflow as tf

from src.master_thesis.data.processing.hashing import ImageHasher
from src.master_thesis.data.processing.augmentation import DataAugmenter
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

    def generate_combined_datasets(self, loader=DatasetLoader()):
            combined_dataset = []
            image_hasher = ImageHasher()

            for dataset_name in os.listdir(self.dataset_dir):
                dataset_path = os.path.join(self.dataset_dir, dataset_name)
                if not os.path.isdir(dataset_path):
                    continue  # Skip non-directory files

                dataset = loader.load(dataset_path)

                if self.augment:
                    dataset = self.augmenter.augment(dataset)

                for image, label in dataset:
                    unique_image = image_hasher.get_image_if_is_not_duplicate(image)

                    if unique_image is not None:
                        combined_dataset.append(
                            (unique_image, label))

            return combined_dataset

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


class DataSetCreator:
    def __init__(self, img_paths, img_labels, image_parser, obscure_images_percent, batch_size):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.image_parser = image_parser
        self.obscure_images_percent = obscure_images_percent
        self.batch_size = batch_size

    def create_datasets(self):
        datasets = {}
        for split in ["train", "val", "test"]:
            datasets[split] = tf.data.Dataset.from_tensor_slices((self.img_paths[split], self.img_labels[split]))
            datasets[split] = datasets[split].map(
                lambda x, y: (self.image_parser(x, self.obscure_images_percent), y),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            if split == "train":
                datasets[split] = datasets[split].shuffle(buffer_size=100)
            datasets[split] = datasets[split].batch(self.batch_size)
            datasets[split] = datasets[split].prefetch(tf.data.experimental.AUTOTUNE)

        return datasets
