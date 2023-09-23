from src.master_thesis.models.results.model_results import ModelResults
import logging

logger = logging.getLogger("ModelEvaluator")


class ModelEvaluator:

    def __init__(self, model_results: ModelResults, metrics=None):
        if metrics is None:
            metrics = ['accuracy']
        self.metrics = metrics
        self.model_results = model_results

    def evaluate_model(self, model, test_dataset):
        """Evaluate a single model's performance on the test dataset."""
        evaluation_results = model.evaluate(test_dataset, metrics=self.metrics)
        return evaluation_results

    def compare_models(self, model_results):
        """Compare the performance of multiple models."""
        # This is a hypothetical method; replace with actual comparison logic
        comparison_results = {}
        for model_name, results in model_results.items():
            for metric in self.metrics:
                comparison_results.setdefault(metric, {})[model_name] = results[metric]
        return comparison_results

    def compare_evaluation_results(self, dataset_name: str):
        dataset_results = self.model_results.get_results_for_dataset(dataset_name)
        for model_name, results in dataset_results.items():
            logger.info(f"Model: {model_name}")
            logger.info(f"Results: {results}\n")
