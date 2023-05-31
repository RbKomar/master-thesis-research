import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pydicom





def parse_dcm_image(img_path):
    img = pydicom.dcmread(img_path).pixel_array
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=-1)  # Add channel dimension
    return img

def load_csv_files(data_path, year):
    df_list = []
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".csv"):
                if "metadata" in file.lower():
                    continue
                if year == 2016:
                    df = pd.read_csv(os.path.join(subdir, file), header=None)
                    df.columns = ['id', 'label']
                    df['label'] = df['label'].replace({'benign': 0, 'malignant': 1})
                else:
                    df = pd.read_csv(os.path.join(subdir, file))
                    df.rename(columns={df.columns[0]: 'id'}, inplace=True)
                    if year in [2017, 2018, 2019]:
                        mel_keyword = 'melanoma' if year == 2017 else 'MEL'
                        df['label'] = df[mel_keyword]
                    else:  # year == 2020
                        df.rename(columns={'target': 'label'}, inplace=True)
                df = df[['id', 'label']]
                df.set_index('id', inplace=True)
                df_list.append(df)
    return pd.concat(df_list)


import matplotlib.pyplot as plt

def plot_label_distribution(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.bar(unique_labels, counts)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Distribution of Labels')
    plt.show()

class Dataset:
    def __init__(self, train, val, test):
        self.train = train
        self.val = val
        self.test = test

def parse_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [256, 256])
    return img

def parse_data(data_path, img_ext, batch_size, image_parser=parse_image, train_prop=0.7, val_prop=0.15, year=2016):
    labels_df = load_csv_files(data_path, year)
    labels_dict = labels_df['label'].to_dict()

    img_paths = {"train": [], "val": [], "test": []}
    img_labels = {"train": [], "val": [], "test": []}  # create a list for labels
    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(img_ext):
                img_id = os.path.splitext(file)[0]
                if img_id in labels_dict:
                    if "val" in subdir.lower():
                        img_paths["val"].append(os.path.join(subdir, file))  # store only paths here
                        img_labels["val"].append(labels_dict[img_id])  # store corresponding labels here
                    elif "test" in subdir.lower():
                        img_paths["test"].append(os.path.join(subdir, file))  # store only paths here
                        img_labels["test"].append(labels_dict[img_id])  # store corresponding labels here
                    else:
                        img_paths["train"].append(os.path.join(subdir, file))  # if not explicitly mentioned use train by default
                        img_labels["train"].append(labels_dict[img_id])

    if len(img_paths["train"]) == 0 or len(img_paths["val"]) == 0 or len(img_paths["test"]) == 0:  # if no folders found, make manual split
        img_paths = img_paths["train"] + img_paths["val"] + img_paths["test"]
        img_labels = img_labels["train"] + img_labels["val"] + img_labels["test"]

        np.random.seed(42)
        indices = np.arange(len(img_paths))  # create a list of indices
        np.random.shuffle(indices)  # shuffle the indices

        total_size = len(img_paths)
        train_idx = int(train_prop * total_size)
        val_idx = int((train_prop + val_prop) * total_size)

        train_img_paths = np.array(img_paths)[indices[:train_idx]]
        train_img_labels = np.array(img_labels)[indices[:train_idx]]

        val_img_paths = np.array(img_paths)[indices[train_idx:val_idx]]
        val_img_labels = np.array(img_labels)[indices[train_idx:val_idx]]

        test_img_paths = np.array(img_paths)[indices[val_idx:]]
        test_img_labels = np.array(img_labels)[indices[val_idx:]]

        img_paths = {"train": train_img_paths, "val": val_img_paths, "test": test_img_paths}
        img_labels = {"train": train_img_labels, "val": val_img_labels, "test": test_img_labels}

    # Create tf.data.Datasets
    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = tf.data.Dataset.from_tensor_slices((img_paths[split], img_labels[split]))
        datasets[split] = datasets[split].map(lambda x, y: (image_parser(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if split == "train":
            datasets[split] = datasets[split].shuffle(buffer_size=100)  # Shuffle the training data before each epoch

        datasets[split] = datasets[split].batch(batch_size)
        datasets[split] = datasets[split].prefetch(tf.data.experimental.AUTOTUNE)

    return Dataset(datasets["train"], datasets["val"], datasets["test"])


def load_isic2016_data(data_path, batch_size):
    return parse_data(data_path, ".jpg", batch_size, year=2016)

def load_isic2017_data(data_path, batch_size):
    return parse_data(data_path, ".jpg", batch_size, year=2017)

def load_isic2018_data(data_path, batch_size):
    return parse_data(data_path, ".jpg", batch_size, year=2018)

def load_isic2019_data(data_path, batch_size):
    return parse_data(data_path, ".jpg", batch_size, year=2019)

def load_isic2020_data(data_path, batch_size):
    return parse_data(data_path, ".jpg", batch_size, year=2020)


def load_dataset(data_path, batch_size):
    if "isic2016" in data_path.lower():
        return load_isic2016_data(data_path, batch_size)
    elif "isic2017" in data_path.lower():
        return load_isic2017_data(data_path, batch_size)
    elif "isic2018" in data_path.lower() or "ham10000" in data_path.lower():
        return load_isic2018_data(data_path, batch_size)
    elif "isic2019" in data_path.lower():
        return load_isic2019_data(data_path, batch_size)
    elif "isic2020" in data_path.lower():
        return load_isic2020_data(data_path, batch_size)
    else:
        raise ValueError(f"Invalid dataset name for: {data_path}")

if __name__ == "__main__":
    from src.data.dataset_generator import DatasetGenerator

    # Initialize dataset generator
    relative_path = os.path.join('..', '..', '..', 'data', 'master-thesis-data')
    absolute_path = os.path.abspath(relative_path)
    print(absolute_path)
    dataset_generator = DatasetGenerator(absolute_path, augment=False, batch_size=32)