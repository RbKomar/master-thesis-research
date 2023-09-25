import pickle

from src.master_thesis.data.generator.dataset_generator import DatasetGenerator
from src.master_thesis.data.handler.dicom_parser import parse_dcm_image
from src.master_thesis.data.handler.hashing import ImageHasher
from src.master_thesis.data.management.dataset_manager import DatasetManager
from src.master_thesis.data.processing.image_processor import ImageParser
from src.master_thesis.data.visualization.visualizer import DataVisualizer


class DataPipeline:
    def __init__(self, dataset_dir, obscure_percent=0, batch_size=32, augment=False):
        """
        Initializes the DataPipeline instance with the provided parameters.
        It also initializes the DatasetGenerator and ImageProcessor instances.
        """
        self.dataset_generator = DatasetGenerator(dataset_dir=dataset_dir, obscure_percent=obscure_percent,
                                                  batch_size=batch_size, augment=augment)
        self.image_processor = ImageParser()
        self.image_hasher = ImageHasher()
        self.datasets = []
        self.visualizer = None

    def load_and_process_data(self, loader=None):
        """
        Loads and processes data using the DatasetGenerator and ImageProcessor instances.
        It also initializes the DataVisualizer instance with the loaded data.
        """
        for dataset_name, dataset in self.dataset_generator.generate_datasets(loader=loader):
            # Processing images in the dataset and managing duplicates using ImageHasher
            for split_name in ['train', 'val', 'test']:
                split_data = getattr(dataset, split_name)
                processed_images = []
                for img_path in split_data['image_path']:
                    # Check if the image is not a duplicate
                    image = self.image_hasher.get_image_if_is_not_duplicate(img_path)
                    if image:
                        # Process the image using ImageProcessor
                        image = self.image_processor.parse_image(img_path, self.dataset_generator.obscure_percent)
                        processed_images.append(image)
                split_data.loc[split_data['image_path'].isin(processed_images), 'image_path'] = processed_images

            # Managing the dataset using Dataset class and adding it to the datasets list
            self.datasets.append(DatasetManager(train=dataset.train, val=dataset.val, test=dataset.test))

            # Initializing the DataVisualizer instance with the loaded data
            if self.visualizer is None:
                self.visualizer = DataVisualizer(data=dataset.train.append([dataset.val, dataset.test]))
            else:
                new_data = dataset.train.append([dataset.val, dataset.test])
                self.visualizer.data = self.visualizer.data.append(new_data, ignore_index=True)

    def visualize_data(self):
        """Visualizes the loaded data using the DataVisualizer instance."""
        if self.visualizer:
            self.visualizer.generate_class_distribution()
            if len(self.datasets) > 1:
                self.visualizer.generate_dataset_comparison([dataset.train.append([dataset.val, dataset.test])
                                                             for dataset in self.datasets])

    def parse_dicom_image(self, img_path):
        """Parses a DICOM image and returns it as a tensor."""
        return parse_dcm_image(img_path)

    def save_pipeline(self, file_path):
        """Saves the DataPipeline instance to a file."""
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
