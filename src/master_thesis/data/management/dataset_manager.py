import tensorflow as tf
from typing import Type


class DatasetManager:
    """
    Class for managing training, validation, and test datasets.
    """

    def __init__(self, train: Type[tf.data.Dataset], val: Type[tf.data.Dataset], test: Type[tf.data.Dataset]):
        self.train = train
        self.val = val
        self.test = test

    def __eq__(self, other):
        return self.train == other.train and self.val == other.val and self.test == other.test

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)

    def __getitem__(self, index):
        if index == 0:
            return self.train
        elif index == 1:
            return self.val
        elif index == 2:
            return self.test
        else:
            raise IndexError("Index out of range. DatasetManager only has 3 subsets: train, val, and test.")

    def get_datasets(self):
        return self.train, self.val, self.test

    def get_train_subset(self, num_samples: int) -> tf.data.Dataset:
        return self.train.take(num_samples)

    def get_val_subset(self, num_samples: int) -> tf.data.Dataset:
        return self.val.take(num_samples)

    def get_test_subset(self, num_samples: int) -> tf.data.Dataset:
        return self.test.take(num_samples)
