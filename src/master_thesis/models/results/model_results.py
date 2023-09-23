import json
import logging

from src.master_thesis.models.train_model import ModelTrainer

logger = logging.getLogger("ModelResults")


class ModelResults:
    """Class responsible for storing and loading model results."""

    def __init__(self):
        self.models_results = {}

    def load_from_json(self, file_path: str):
        try:
            with open(file_path, 'r') as file:
                self.models_results = json.load(file)
            if not validate_results_structure(self.models_results):
                logger.error(f"Invalid structure in {file_path}.")
                raise ValueError(f"Invalid structure in {file_path}.")
        except FileNotFoundError:
            logger.error(f"File {file_path} not found.")
            raise FileNotFoundError(f"File {file_path} not found.")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {file_path}.")
            raise json.JSONDecodeError(f"Failed to decode JSON from {file_path}.", file_path, 0)
    def add_model_result(self, model_name: str, model_trainer: ModelTrainer):
        """Add results of a model to the internal dictionary."""
        self.models_results[model_name] = {
            'history': model_trainer.history,
            'train_time': model_trainer.train_time,
            'inference_time': model_trainer.inference_time,
            'predictions': model_trainer.predictions
        }

    def get_results_for_dataset(self, dataset_name: str) -> dict:
        """Extract results for a specific dataset."""
        dataset_results = {}
        for model_name, results in self.models_results.items():
            if dataset_name in results:
                dataset_results[model_name] = results[dataset_name]
        return dataset_results


def validate_results_structure(results: dict) -> bool:
    """Validate the structure of the loaded model results."""
    for model_name, model_data in results.items():
        if 'history' not in model_data:
            logger.error(f"'history' key missing for model {model_name}.")
            return False
        if 'binary_accuracy' not in model_data['history']:
            logger.error(f"'binary_accuracy' key missing in 'history' for model {model_name}.")
            return False
    return True


class MockModelResults(ModelResults):
    """A mock class to simulate ModelResults."""

    def get_results_for_dataset(self, dataset_name: str):
        """Return mock results for the given dataset."""
        if dataset_name == 'dataset1':
            return {
                'model1': {
                    'history': {
                        'binary_accuracy': [0.7, 0.8, 0.85, 0.9],
                        'val_binary_accuracy': [0.7, 0.75, 0.8, 0.85],
                        'auc': [0.7, 0.8, 0.85, 0.9],
                        'val_auc': [0.7, 0.75, 0.8, 0.85],
                        'f1_score': [0.6, 0.7, 0.75, 0.8],
                        'val_f1_score': [0.6, 0.65, 0.7, 0.75],
                        'loss': [0.6, 0.5, 0.4, 0.3],
                        'val_loss': [0.6, 0.55, 0.5, 0.45]
                    }
                }
            }
        else:
            return {}
