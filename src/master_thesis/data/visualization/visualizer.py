import plotly.express as px
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


class DataVisualizer:
    def __init__(self, data: pd.DataFrame):
        self.data = data  # DataFrame containing metadata about images

    def generate_class_distribution(self) -> px.Figure:
        fig = px.histogram(self.data, x='label')
        fig.show()
        return fig

    @staticmethod
    def generate_dataset_comparison(datasets: List[pd.DataFrame]) -> px.Figure:
        combined_data = pd.concat(datasets, keys=[f'Dataset {i}' for i in range(len(datasets))], names=['Dataset'])
        combined_data = combined_data.reset_index(level='Dataset').reset_index(drop=True)

        fig = px.histogram(combined_data, x='label', color='Dataset')
        fig.show()
        return fig

    def display_sample_images(self, dataset_name: str, num_images_per_class: int = 1):
        # Filter data for the specified dataset
        dataset_data = self.data[self.data['dataset'] == dataset_name]

        # Get unique classes in the dataset
        classes = dataset_data['label'].unique()

        # For each class, display sample images
        for label in classes:
            class_data = dataset_data[dataset_data['label'] == label]
            sample_images = class_data.sample(n=num_images_per_class)['image_path'].tolist()

            # Display images
            for i, img_path in enumerate(sample_images, 1):
                img = mpimg.imread(img_path)
                plt.subplot(len(classes), num_images_per_class, i)
                plt.imshow(img)
                plt.axis('off')
                plt.title(label)
            plt.show()


import numpy as np
import matplotlib.pyplot as plt


def plot_label_distribution(labels):
    """Plot the distribution of labels in the dataset."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.bar(unique_labels, counts)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Distribution of Labels')
    plt.show()
