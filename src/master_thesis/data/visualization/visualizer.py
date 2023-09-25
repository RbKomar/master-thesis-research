import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import List, Optional
import logging

logger = logging.getLogger("DataVisualizer")


class DataVisualizer:
    """
    Class for generating various visualizations for data analysis.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataVisualizer with a DataFrame containing metadata about images.

        Args:
            data (pd.DataFrame): DataFrame containing metadata about images.
        """
        self.data = data

    def generate_class_distribution(self) -> Optional[px.Figure]:
        """
        Generate and display a histogram showing the distribution of classes in the data.

        Returns:
            Optional[px.Figure]: A plotly express Figure object representing the histogram, or None if an error occurred during generation.
        """
        try:
            fig = px.histogram(self.data, x='label')
            fig.show()
            return fig
        except Exception as e:
            logger.error(f"Error generating class distribution visualization: {e}", exc_info=True)
            return None

    @staticmethod
    def generate_dataset_comparison(datasets: List[pd.DataFrame]) -> Optional[px.Figure]:
        """
        Generate and display a histogram comparing the class distribution among different datasets.

        Args:
            datasets (List[pd.DataFrame]): A list of DataFrames, each representing a different dataset.

        Returns:
            Optional[px.Figure]: A plotly express Figure object representing the histogram, or None if an error occurred during generation.
        """
        try:
            combined_data = pd.concat(datasets, keys=[f'Dataset {i}' for i in range(len(datasets))], names=['Dataset'])
            combined_data = combined_data.reset_index(level='Dataset').reset_index(drop=True)
            fig = px.histogram(combined_data, x='label', color='Dataset')
            fig.show()
            return fig
        except Exception as e:
            logger.error(f"Error generating dataset comparison visualization: {e}", exc_info=True)
            return None

    def display_sample_images(self, dataset_name: str, num_images_per_class: int = 1) -> None:
        """
        Display sample images for each class from a specified dataset.

        Args:
            dataset_name (str): The name of the dataset from which to display sample images.
            num_images_per_class (int, optional): The number of images to display per class. Defaults to 1.
        """
        try:
            dataset_data = self.data[self.data['dataset'] == dataset_name]
            classes = dataset_data['label'].unique()
            for label in classes:
                class_data = dataset_data[dataset_data['label'] == label]
                sample_images = class_data.sample(n=num_images_per_class)['image_path'].tolist()
                for i, img_path in enumerate(sample_images, 1):
                    img = mpimg.imread(img_path)
                    plt.subplot(len(classes), num_images_per_class, i)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(label)
                plt.show()
        except Exception as e:
            logger.error(f"Error displaying sample images for dataset {dataset_name}: {e}", exc_info=True)


def plot_label_distribution(labels: np.ndarray) -> None:
    """
    Plot the distribution of labels in the dataset.

    Args:
        labels (np.ndarray): An array containing the labels in the dataset.
    """
    try:
        unique_labels, counts = np.unique(labels, return_counts=True)
        plt.bar(unique_labels, counts)
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Distribution of Labels')
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting label distribution: {e}", exc_info=True)
