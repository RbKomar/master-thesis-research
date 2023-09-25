import json

from src.master_thesis.models.train_model import ModelTrainer
import logging

logger = logging.getLogger("ModelResults")


class ModelResults:
    """Class responsible for storing and loading model results."""

    def __init__(self):
        """
        Initialize ModelResults instance with an empty dictionary for model results.
        """
        self.models_results = {}

    def load_from_json(self, file_path: str):
        """
        Load model results from a JSON file and validate the structure.

        :param file_path: str, Path to the JSON file containing model results.
        """
        try:
            with open(file_path, 'r') as file:
                self.models_results = json.load(file)
            if not self.validate_results_structure():
                logger.error(
                    f"Invalid structure in {file_path}. Expected keys are 'history', 'binary_accuracy' in 'history'.")
                raise ValueError(f"Invalid structure in {file_path}.")
        except FileNotFoundError:
            logger.error(f"File {file_path} not found.")
            raise FileNotFoundError(f"File {file_path} not found.")
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {file_path}.")
            raise json.JSONDecodeError(f"Failed to decode JSON from {file_path}.")

    def add_model_result(self, model_name: str, model_trainer: ModelTrainer):
        """
        Add results of a model to the internal dictionary.

        :param model_name: str, The name of the model.
        :param model_trainer: ModelTrainer, An instance of the ModelTrainer class containing model results.
        """
        self.models_results[model_name] = {
            'history': model_trainer.history,
            'train_time': model_trainer.train_time,
            'inference_time': model_trainer.inference_time,
            'predictions': model_trainer.predictions
        }

    def get_results_for_dataset(self, dataset_name: str) -> dict:
        """
        Extract results for a specific dataset from the stored model results.

        :param dataset_name: str, The name of the dataset.
        :return: dict, A dictionary containing results for the specified dataset.
        """
        dataset_results = {}
        for model_name, results in self.models_results.items():
            if dataset_name in results:
                dataset_results[model_name] = results[dataset_name]
        if not dataset_results:
            logger.warning(f"No results found for dataset: {dataset_name}")
        return dataset_results

    def validate_results_structure(self) -> bool:
        """
        Validate the structure of the loaded model results.

        :return: bool, True if the structure is valid, False otherwise.
        """
        for model_name, model_data in self.models_results.items():
            if 'history' not in model_data:
                logger.error(f"'history' key missing for model {model_name}.")
                return False
            if 'binary_accuracy' not in model_data['history']:
                logger.error(f"'binary_accuracy' key missing in 'history' for model {model_name}.")
                return False
        return True
