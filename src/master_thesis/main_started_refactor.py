import logging

from src.master_thesis.pipelines.pipeline import IntegratedPipeline


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MainRoutine:

    def __init__(self, base_config, fraction=1.0):
        self.logger = logging.getLogger("MainRoutine")
        self.pipeline = IntegratedPipeline(base_config)
        self.fraction = fraction

    def _slice_data(self, dataset):
        """Retrieve a fraction of the dataset for testing purposes."""
        slice_index = int(len(dataset) * self.fraction)
        return dataset[:slice_index]

    def load_process_and_visualize_data(self):
        try:
            self.pipeline.load_process_and_visualize_data()
        except Exception as e:
            self.logger.error(f"Error in data loading and processing: {str(e)}", exc_info=True)

    def create_train_and_evaluate_models(self, train_dataset, test_dataset=None):
        train_dataset = self._slice_data(train_dataset)
        if test_dataset:
            test_dataset = self._slice_data(test_dataset)

        try:
            self.pipeline.create_train_and_evaluate_models(train_dataset, test_dataset)
        except Exception as e:
            self.logger.error(f"Error in model creation and evaluation: {str(e)}", exc_info=True)

    def save_results(self, results_file_path):
        self.pipeline.save_results(results_file_path)


def main():
    setup_logging()
    fraction_for_testing = 0.01  # using only 1% of the dataset for testing
    base_config = {
        'environment': 'development',
        'data_config': {
            'dataset_dir': ''
        },
        'model_config': {}
    }

    routine = MainRoutine(base_config, fraction_for_testing)
    routine.load_process_and_visualize_data()

    for dataset_name, dataset in routine.pipeline.data_pipeline.dataset_generator.generate_datasets():
        routine.create_train_and_evaluate_models(dataset.train, dataset.test)

    combined_dataset = routine.pipeline.data_pipeline.dataset_generator.generate_combined_datasets()
    routine.create_train_and_evaluate_models(combined_dataset)

    results_file_path = "results_comparison.json"
    routine.save_results(results_file_path)


if __name__ == "__main__":
    main()
