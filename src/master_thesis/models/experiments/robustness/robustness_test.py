from src.master_thesis.augmentation import DataAugmenter
from src.master_thesis.data.generator.dataset_generator import DatasetGenerator
from src.master_thesis.data.processing.image_processor import ImageProcessor
from plotting import PerformanceVisualization  # Placeholder for visualization logic

from src.master_thesis.models.evaluation.evaluator import ModelEvaluator
from src.master_thesis.models.train_model import ModelHandler


class RobustnessTester:

    def __init__(self, model: ModelHandler, evaluator: ModelEvaluator, dataset_path,
                 augmenter: DataAugmenter, generator: DatasetGenerator):
        self.dataset_path = dataset_path
        self.augmenter = augmenter
        self.generator = generator
        self.image_processor = ImageProcessor()
        self.model = model
        self.evaluator = evaluator

    def generate_obscured_dataset(self, occlusion_size=(50, 50), noise_level=0.05):
        test_dataset = self.generator.generate_combined_datasets()

        # Apply occlusions and noise to the images
        obscured_dataset = test_dataset.map(lambda x, y:
                                            (self.image_processor.occlude_image(x, occlusion_size),
                                             self.image_processor.add_noise(x, noise_level), y))
        return obscured_dataset

    def generate_incomplete_dataset(self, removal_fraction=0.3):
        test_dataset = self.generator.generate_combined_datasets()

        # Remove sections of the images
        incomplete_dataset = test_dataset.map(lambda x, y:
                                              (self.image_processor.remove_section(x, removal_fraction), y))
        return incomplete_dataset

    def train_model(self, training_data):
        """Train the model using the provided training data."""
        self.model.train(training_data)

    def evaluate_robustness(self, original_test_data):
        """Evaluate the model on various test datasets and return the metrics."""
        metrics = {}

        # Evaluate on original test data
        metrics["original"] = self.evaluator.evaluate(self.model, original_test_data)

        # Evaluate on obscured test data
        obscured_test_data = self.generate_obscured_dataset()
        metrics["obscured"] = self.evaluator.evaluate(self.model, obscured_test_data)

        # Evaluate on incomplete test data
        incomplete_test_data = self.generate_incomplete_dataset()
        metrics["incomplete"] = self.evaluator.evaluate(self.model, incomplete_test_data)

        return metrics

    @staticmethod
    def visualize_results(metrics):
        """Visualize the evaluation metrics."""
        # Placeholder for visualization logic
        PerformanceVisualization.plot_metrics(metrics)
