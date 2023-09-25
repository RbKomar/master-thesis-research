import os
import numpy as np


class ImagePathProcessor:
    def __init__(self, labels_dict, img_ext):
        self.labels_dict = labels_dict
        self.img_ext = img_ext

    def process_paths(self, data_path, train_prop, val_prop):
        img_paths, img_labels = self._process_paths(data_path)
        return self._split_data(img_paths, img_labels, train_prop, val_prop)

    def _process_paths(self, data_path):
        img_paths = {"train": [], "val": [], "test": []}
        img_labels = {"train": [], "val": [], "test": []}
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(self.img_ext):
                    img_id = os.path.splitext(file)[0]
                    if img_id in self.labels_dict:
                        label = self.labels_dict[img_id]
                        split = self._determine_split(subdir)
                        img_paths[split].append(os.path.join(subdir, file))
                        img_labels[split].append(label)
        return img_paths, img_labels

    @staticmethod
    def _determine_split(subdir):
        if "val" in subdir.lower():
            return "val"
        elif "test" in subdir.lower():
            return "test"
        else:
            return "train"

    def _split_data(self, img_paths, img_labels, train_prop, val_prop):
        np.random.seed(42)
        indices = np.arange(len(img_paths["train"]))

        total_size = len(img_paths["train"])
        train_idx = int(train_prop * total_size)
        val_idx = int((train_prop + val_prop) * total_size)

        train_img_paths, val_img_paths, test_img_paths = self._split_paths(img_paths["train"], indices, train_idx,
                                                                           val_idx)
        train_img_labels, val_img_labels, test_img_labels = self._split_paths(img_labels["train"], indices, train_idx,
                                                                              val_idx)

        img_paths["train"], img_paths["val"], img_paths["test"] = train_img_paths, val_img_paths, test_img_paths
        img_labels["train"], img_labels["val"], img_labels["test"] = train_img_labels, val_img_labels, test_img_labels

        return img_paths, img_labels

    @staticmethod
    def _split_paths(paths, indices, train_idx, val_idx):
        np.random.shuffle(indices)
        train_paths = np.array(paths)[indices[:train_idx]]
        val_paths = np.array(paths)[indices[train_idx:val_idx]]
        test_paths = np.array(paths)[indices[val_idx:]]
        return train_paths, val_paths, test_paths
