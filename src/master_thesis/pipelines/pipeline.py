import logging
import pickle

from src.master_thesis.config.config import ConfigManager
from src.master_thesis.models.model_factory import ModelFactory
from src.master_thesis.pipelines.data_pipeline.data_pipeline import DataPipeline
from src.master_thesis.pipelines.model_pipeline.model_pipeline import ModelPipeline


class IntegratedPipeline:
    def __init__(self, base_config=None):
        self.logger = logging.getLogger("IntegratedPipeline")

        if not base_config:
            base_config = {
                'environment': 'development',
                'data_config': {},
                'model_config': {}
            }
        self.config_manager = ConfigManager(**base_config)

        self.data_pipeline = DataPipeline(self.config_manager.data_config)

        model_factory = ModelFactory()
        self.model_pipeline = ModelPipeline(self.config_manager.model_config, model_factory)

    def load_process_and_visualize_data(self, loader=None, dataset_names=None):
        """Load, process, and visualize data."""
        try:
            self.data_pipeline.load_and_process_data(loader=loader)
            self.data_pipeline.visualize_data(dataset_names=dataset_names)
            self.logger.info("Data loading, processing, and visualization completed.")
        except Exception as e:
            self.logger.error(f"Error in data processing: {str(e)}", exc_info=True)

    def create_train_and_evaluate_models(self, train_dataset, test_dataset=None):
        """Create, train, and evaluate models."""
        try:
            self.model_pipeline.create_models()
            self.model_pipeline.process_models(train_dataset, test_dataset)
            self.logger.info("Models created, trained, and evaluated.")
        except Exception as e:
            self.logger.error(f"Error in model processing: {str(e)}", exc_info=True)

    def save_results(self, file_path):
        """Save results to a specified file."""
        try:
            self.model_pipeline.save_results_to_file(file_path)
            self.logger.info(f"Results saved to {file_path}.")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}", exc_info=True)

    def update_config(self, new_config: dict):
        """Allows updating configurations even after initializing."""
        if 'data_config' in new_config:
            self.config_manager.data_config.update(new_config['data_config'])
        if 'model_config' in new_config:
            self.config_manager.model_config.update(new_config['model_config'])
        self.logger.info("Configuration updated.")

    def run_pipeline(self, loader=None, dataset_names=None, train_dataset=None, test_dataset=None,
                     results_file_path=None):
        """Runs the entire pipeline from loading data to saving results."""
        self.logger.info("Pipeline started.")

        self.load_process_and_visualize_data(loader=loader, dataset_names=dataset_names)
        self.create_train_and_evaluate_models(train_dataset, test_dataset)

        if results_file_path:
            self.save_results(results_file_path)

        self.logger.info("Pipeline completed.")

    def save_pipeline_state(self, file_path: str):
        """Saves the state of the pipeline."""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            self.logger.info(f"Pipeline state saved to {file_path}.")
        except Exception as e:
            self.logger.error(f"Error saving pipeline state: {str(e)}", exc_info=True)

    @staticmethod
    def load_pipeline_state(file_path: str):
        """Load a saved pipeline state."""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.getLogger("IntegratedPipeline").error(f"Error loading pipeline state: {str(e)}", exc_info=True)
            return None
