# src/data/dataset_generator.py
import os
import json

import cv2

from .make_dataset import load_dataset, load_isic2016_data, load_isic2017_data, load_isic2018_data, load_isic2019_data, \
    load_isic2020_data
from .augmentation import DataAugmenter
import hashlib
from PIL import Image
import numpy as np


def get_image_hash(image_path):
    """Generate a hash for an image."""
    image = Image.open(image_path)
    image_data = np.asarray(image)
    return hashlib.md5(image_data.tobytes()).hexdigest()


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
        image_hashes = set()
        combined_dataset_name = "_combined"
        combined_dataset = {}

        for dataset_name in self.dataset_names:
            if "isic2016" in dataset_name.lower():
                dataset, csv_files_ground_truth = load_isic2016_data(os.path.join(self.dataset_dir, dataset_name),
                                                                     self.batch_size)
            elif "isic2017" in dataset_name.lower():
                dataset = load_isic2017_data(os.path.join(self.dataset_dir, dataset_name), self.batch_size)
            elif "isic2018" in dataset_name.lower():
                dataset = load_isic2018_data(os.path.join(self.dataset_dir, dataset_name), self.batch_size)
            elif "isic2019" in dataset_name.lower():
                dataset = load_isic2019_data(os.path.join(self.dataset_dir, dataset_name), self.batch_size)
            elif "isic2020" in dataset_name.lower():
                dataset = load_isic2020_data(os.path.join(self.dataset_dir, dataset_name), self.batch_size)

            if self.augment:
                dataset = self.data_augmentation.augment(dataset)

            # Remove duplicates based on image hash
            dataset = [image for image in dataset if get_image_hash(image) not in image_hashes]

            # Add hashes of current dataset to the set of all hashes
            image_hashes.update(get_image_hash(image) for image in dataset)

            if not combined_dataset:
                combined_dataset = dataset
            else:
                combined_dataset = np.concatenate([combined_dataset, dataset])

        yield combined_dataset_name + ("_augmented" if self.augment else ""), combined_dataset

    def generate_datasets(self):
        for dataset_name in self.dataset_names:
            dataset = load_dataset(os.path.join(self.dataset_dir, dataset_name), self.batch_size)
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
