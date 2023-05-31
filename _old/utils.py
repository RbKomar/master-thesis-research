# utils.py
import os
import numpy as np

def load_data(dataset_dir: str):
    x_train = np.load(os.path.join(dataset_dir, 'train_data.npy'))
    y_train = np.load(os.path.join(dataset_dir, 'train_labels.npy'))
    x_val = np.load(os.path.join(dataset_dir, 'val_data.npy'))
    y_val = np.load(os.path.join(dataset_dir, 'val_labels.npy'))
    x_test = np.load(os.path.join(dataset_dir, 'test_data.npy'))
    y_test = np.load(os.path.join(dataset_dir, 'test_labels.npy'))

    return x_train, y_train, x_val, y_val, x_test, y_test
