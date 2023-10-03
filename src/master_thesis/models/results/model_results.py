import json
import os
from typing import Dict
from src.master_thesis.models.train_model import ModelHandler
import logging
logger = logging.getLogger("ModelResults")


class ModelResults:
    """Class responsible for storing and loading model results."""

    def __init__(self):
        self.models_results = {}
        self.logger = logger
        self.logger.info("ModelResults object has been initialized.")

    def _validate_results_structure(self) -> bool:
        required_keys = ['history', 'train_time', 'eval_time', 'n_params', 'evaluation_metrics', 'predictions']
        for model_name, datasets in self.models_results.items():
            for dataset_name, results in datasets.items():
                if any(key not in results for key in required_keys):
                    self.logger.error(f"Missing keys for model {model_name} on dataset {dataset_name}.")
                    return False
        return True

    @staticmethod
    def _validate_single_result(results: Dict) -> bool:
        required_keys = ['history', 'train_time', 'eval_time', 'n_params', 'evaluation_metrics', 'predictions']
        return all(key in results for key in required_keys)

    def add_model_result(self, model_name: str, dataset_name: str, results: Dict):
        if not self._validate_single_result(results):
            self.logger.error(f"Invalid result structure for model {model_name} on dataset {dataset_name}.")
            raise ValueError(f"Invalid result structure for model {model_name} on dataset {dataset_name}.")
        self.models_results.setdefault(model_name, {})[dataset_name] = results
        self.logger.info(f"Results for model {model_name} on dataset {dataset_name} have been added.")

    def load_from_json(self, file_path: str):
        """
        Load model results from a JSON file and validate the structure.

        :param file_path: str, Path to the JSON file containing model results.
        """
        try:
            with open(file_path, 'r') as file:
                self.models_results = json.load(file)
            if not self._validate_results_structure():
                self.logger.error(f"Invalid structure in loaded results from {file_path}.")
                raise ValueError(f"Invalid structure in loaded results from {file_path}.")
        except FileNotFoundError:
            logger.error(f"File {file_path} not found.")
            raise FileNotFoundError(f"File {file_path} not found.")

    def get_results_for_dataset(self, dataset_name: str) -> Dict[str, Dict]:
        """
        Extract results for a specific dataset from the stored model results.

        :param dataset_name: str, The name of the dataset.
        :return: dict, A dictionary containing results for the specified dataset.
        """
        if not dataset_name:
            self.logger.warning("Dataset name is not provided.")
            raise ValueError("Dataset name cannot be None or an empty string.")
        dataset_results = {}
        for model_name, results in self.models_results.items():
            if dataset_name in results:
                dataset_results[model_name] = results[dataset_name]
        if not dataset_results:
            logger.warning(f"No results found for dataset: {dataset_name}")
        return dataset_results

    def save_to_json(self, file_path: str):
        """
        Save model results to a JSON file.

        :param file_path: str, Path to the JSON file to save model results.
        """
        try:
            # Check if the directory exists, if not create it
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(file_path, 'w') as file:
                json.dump(self.models_results, file)
            logger.info(f"Model results successfully saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save model results to {file_path} due to {str(e)}")
            raise e
