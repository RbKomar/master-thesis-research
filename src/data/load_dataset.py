import os
import tensorflow as tf
import pandas as pd

def load_ham10000(data_dir):
    metadata_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    metadata_df = pd.read_csv(metadata_path)
    image_dir = os.path.join(data_dir, 'HAM10000_images_part_1')
    image_filenames = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]
    labels = metadata_df['dx'].tolist()
    labels_dict = {label: i for i, label in enumerate(set(labels))}
    labels = [labels_dict[label] for label in labels]
    return tf.data.Dataset.from_tensor_slices((image_filenames, labels))

def load_isic2016(data_dir):
    metadata_path = os.path.join(data_dir, 'ISIC-2016_Training_Part3_GroundTruth.csv')
    metadata_df = pd.read_csv(metadata_path)
    image_dir = os.path.join(data_dir, 'ISIC-2016_Training_Data')
    image_filenames = [os.path.join(image_dir, fname + '.jpg') for fname in metadata_df['image_id'].tolist()]
    labels = metadata_df['melanoma'].tolist()
    return tf.data.Dataset.from_tensor_slices((image_filenames, labels))

def load_isic2017(data_dir):
    metadata_path = os.path.join(data_dir, 'ISIC-2017_Training_Part3_GroundTruth.csv')
    metadata_df = pd.read_csv(metadata_path)
    image_dir = os.path.join(data_dir, 'ISIC-2017_Training_Data')
    image_filenames = [os.path.join(image_dir, fname + '.jpg') for fname in metadata_df['image'].tolist()]
    labels = metadata_df['melanoma'].tolist()
    return tf.data.Dataset.from_tensor_slices((image_filenames, labels))

def load_isic2019(data_dir):
    metadata_path = os.path.join(data_dir, 'ISIC_2019_Training_Metadata.csv')
    metadata_df = pd.read_csv(metadata_path)
    image_dir = os.path.join(data_dir, 'ISIC_2019_Training_Input')
    image_filenames = [os.path.join(image_dir, fname + '.jpg') for fname in metadata_df['image'].tolist()]
    labels = metadata_df['melanoma'].tolist()
    return tf.data.Dataset.from_tensor_slices((image_filenames, labels))

def load_isic2020(data_dir):
    metadata_path = os.path.join(data_dir, 'ISIC_2020_Training_GroundTruth.csv')
    metadata_df = pd.read_csv(metadata_path)
    image_dir = os.path.join(data_dir, 'ISIC_2020_Training_Input')
    image_filenames = [os.path.join(image_dir, fname + '.jpg') for fname in metadata_df['image_name'].tolist()]
    labels = metadata_df['benign_malignant'].tolist()
    return tf.data.Dataset.from_tensor_slices((image_filenames, labels))

def load_dataset(data_dir):
    dataset_name = os.path.basename(data_dir)
    if dataset_name == 'ham10000':
        return load_ham10000(data_dir)
    elif dataset_name == 'isic2016':
        return load_isic2016(data_dir)
    elif dataset_name == 'isic2017':
        return load_isic2017(data_dir)
    elif dataset_name == 'isic2019':
        return load_isic2019(data_dir)
    elif dataset_name == 'isic2020':
        return load_isic2020(data_dir)
