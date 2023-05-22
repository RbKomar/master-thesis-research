# -*- coding: utf-8 -*-
import argparse
import os

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.data.load_dataset import load_ham10000, load_isic2016, load_isic2017, load_isic2019, load_isic2020
from src.data.preprocessing import preprocess_image

logger = logging.getLogger("master-thesis")


import hashlib

def hash_image(image):
    """Compute a hash of the image data."""
    hasher = hashlib.sha256()
    hasher.update(image.tobytes())
    return hasher.digest()

def make_dataset(data_dir, image_size, batch_size, shuffle_buffer_size):
    datasets = []
    all_hashes = set() # set of all image hashes
    dataset_info = [
        {"name": "HAM10000", "load_func": load_ham10000, "preprocess_func": preprocess_image},
        {"name": "ISIC2016", "load_func": load_isic2016, "preprocess_func": preprocess_image},
        {"name": "ISIC2017", "load_func": load_isic2017, "preprocess_func": preprocess_image},
        {"name": "ISIC2019", "load_func": load_isic2019, "preprocess_func": preprocess_image},
        {"name": "ISIC2020", "load_func": load_isic2020, "preprocess_func": preprocess_image}
    ]
    for info in dataset_info:
        # load the dataset
        data = info["load_func"](data_dir)
        # remove duplicates
        data = data.filter(lambda image: hash_image(image) not in all_hashes)
        # add new hashes to set
        all_hashes.update(set(data.map(lambda image: hash_image(image))))
        # preprocess the images
        data = data.map(lambda image, label: (info["preprocess_func"](image, image_size), label))
        # cache the dataset
        data = data.cache()
        # shuffle the dataset
        data = data.shuffle(buffer_size=shuffle_buffer_size)
        # batch the dataset
        data = data.batch(batch_size)
        datasets.append(data)
        print(f"{info['name']} dataset loaded with {len(data)} images.")
    return tf.data.experimental.sample_from_datasets(datasets)


def save_data(data, filename):
    """
    Save the specified data to a CSV file.

    Args:
        data (pd.DataFrame): Data to be saved to a CSV file.
        filename (str): Path to the CSV file to save.
    """
    data.to_csv(filename, index=False)

def cache_dataset(dataset, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    for i, image in enumerate(dataset):
        cache_path = os.path.join(cache_dir, f'image_{i}.npy')
        np.save(cache_path, image.numpy())
    return tf.data.Dataset.list_files(os.path.join(cache_dir, '*.npy'))


def split_dataset(dataset, train_ratio, val_ratio, test_ratio):
    num_examples = len(dataset)
    train_size = int(train_ratio * num_examples)
    val_size = int(val_ratio * num_examples)
    test_size = int(test_ratio * num_examples)
    assert train_size + val_size + test_size == num_examples, 'Invalid split ratios'
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size+val_size]
    test_dataset = dataset[train_size+val_size:]
    return train_dataset, val_dataset, test_dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and save HAM10000 dataset')
    parser.add_argument('--data_dir', type=str, default=os.environ.get('DATASET_DIR', './data'),
                        help='Path to directory containing the HAM10000 dataset')
    parser.add_argument('--version', type=str, default='v1',
                        help='Version number or date of the dataset')
    args = parser.parse_args()

    train_data, test_data = load_data(args.data_dir, args.version)

    train_df = pd.DataFrame({'image_path': train_data.filepaths, 'class': train_data.labels})
    test_df = pd.DataFrame({'image_path': test_data.filepaths, 'class': test_data.labels})

    save_data(train_df, os.path.join(args.data_dir, 'train.csv'))
    save_data(test_df, os.path.join(args.data_dir, 'test.csv'))

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    # Usage example:
    data_dir = 'path/to/data'
    cache_dir = 'path/to/cache'

    # Load and preprocess the HAM10000 dataset
    dataset = load_ham10000(data_dir)

    # Cache the preprocessed images
    cached_dataset = cache_dataset(dataset, cache_dir)

    # Use the cached dataset for training
    train_dataset = cached_dataset.take(8000).shuffle(8000).batch(batch_size)
    val_dataset = cached_dataset.skip(8000).take(1000).batch(batch_size)
    test_dataset = cached_dataset.skip(9000).batch(batch_size)

