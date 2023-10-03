from src.master_thesis.models.model_factory import ModelFactory
from src.master_thesis.models.train_model import ModelHandler
from src.master_thesis.models.results.model_results import ModelResults
import logging


class ModelPipeline:
    def __init__(self, model_config: dict, model_factory: ModelFactory, results_manager: ModelResults = None):
        self.model_factory = model_factory
        self.model_config = model_config
        self.models = []
        self.logger = logging.getLogger("ModelPipeline")
        self.results_manager = results_manager if results_manager else ModelResults()

    def create_models(self):
        model_handlers = self.model_factory.model_handler_mapping.keys()

        for identifier in model_handlers:
            self._create_single_model(identifier)

    def _create_single_model(self, identifier: str):
        try:
            model_handler = self.model_factory.create_model_handler(identifier, **self.model_config)
            self.models.append(model_handler)
        except Exception as e:
            self.logger.error(f"Error while creating {identifier}: {str(e)}")

    def process_models(self, train_dataset, test_dataset=None):
        for model in self.models:
            self._train_single_model(model, train_dataset)

            if test_dataset:
                results = self._evaluate_single_model(model, test_dataset)
                self._store_results(model, test_dataset.name, results)

    def _train_single_model(self, model: ModelHandler, dataset):
        try:
            model.train(train_dataset)
        except Exception as e:
            self.logger.error(f"Error while training model {type(model).__name__}: {str(e)}")

    def _evaluate_single_model(self, model: ModelHandler, test_dataset):
        try:
            results = model.evaluate(test_dataset)  # Assuming the model_handler's 'evaluate' method returns results.
            return results
        except Exception as e:
            self.logger.error(f"Error while evaluating model {type(model).__name__}: {str(e)}")
            return None

    def _store_results(self, model: ModelHandler, dataset_name: str, results: dict):
        model_name = type(model).__name__
        if results:
            self.results_manager.add_model_result(model_name, dataset_name, results)

    def save_results_to_file(self, file_path: str):
        self.results_manager.save_to_json(file_path)

    def clear_models(self):
        """Clear the models stored in the pipeline."""
        self.models = []

