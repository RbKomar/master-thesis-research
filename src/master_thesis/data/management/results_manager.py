import json
import os
import glob


class ResultsManager:
    def __init__(self, results_dir):
        self.results_dir = results_dir

    def save_results(self, model_name, dataset_name, results, epochs):
        results_filename = f'results_epochs_{epochs}.json'
        filepath = os.path.join(self.results_dir, results_filename)

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        existing_data = {}
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                existing_data = json.load(file)

        existing_data.setdefault(model_name, {})[dataset_name] = results

        with open(filepath, 'w+') as file:
            json.dump(existing_data, file)

    # Example: manager.fetch_results(model_name='model_1', epochs=50)
    def fetch_results(self, model_name=None, dataset_name=None, epochs=None):
        results_filename = f'results_epochs_{epochs}.json' if epochs else '*.json'
        filepath = os.path.join(self.results_dir, results_filename)

        results = {}
        for file in glob.glob(filepath):
            with open(file, 'r') as f:
                data = json.load(f)
                if model_name:
                    data = {model_name: data.get(model_name, {})}
                    if dataset_name:
                        data[model_name] = {dataset_name: data[model_name].get(dataset_name, {})}
                results.update(data)

        return results

    # Example: manager.compare_results(model_name='model_1')
    def compare_results(self, *args, **kwargs):
        results = self.fetch_results(*args, **kwargs)

        comparison_summary = {}
        for model_name, datasets_results in results.items():
            for dataset_name, dataset_result in datasets_results.items():
                comparison_key = f"{model_name}_{dataset_name}"
                # Here you need to implement your comparison logic based on your result structure
                # comparison_summary[comparison_key] = comparison of dataset_result

        return comparison_summary

    # Example: manager.filter_results(results, {'accuracy': 0.9})
    @staticmethod
    def filter_results(results, filter_criteria):
        filtered_results = {}
        for model_name, datasets_results in results.items():
            for dataset_name, dataset_result in datasets_results.items():
                meets_criteria = all(dataset_result.get(key) == value for key, value in filter_criteria.items())
                if meets_criteria:
                    filtered_results.setdefault(model_name, {})[dataset_name] = dataset_result

        return filtered_results

    # Example: manager.aggregate_results(results, {'accuracy': np.mean})
    @staticmethod
    def aggregate_results(results, aggregation_criteria):
        aggregated_results = {}
        for model_name, datasets_results in results.items():
            for dataset_name, dataset_result in datasets_results.items():
                for key, aggregation_function in aggregation_criteria.items():
                    agg_key = f"{key}_{aggregation_function.__name__}"
                    value_list = [res[key] for res in dataset_result if key in res]
                    aggregated_results.setdefault(model_name, {}).setdefault(dataset_name, {})[
                        agg_key] = aggregation_function(value_list) if value_list else None

        return aggregated_results
