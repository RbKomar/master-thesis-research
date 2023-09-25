import glob
import os
import json


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

        if model_name not in existing_data:
            existing_data[model_name] = {}

        existing_data[model_name][dataset_name] = results

        with open(filepath, 'w+') as file:
            json.dump(existing_data, file)


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

    def compare_results(self, *args, **kwargs):
        results = self.fetch_results(*args, **kwargs)
        # Perform comparison and generate a summary of the results

    def filter_results(self, results, filter_criteria):
        # Apply filter criteria to results and return filtered results.
        pass

    def aggregate_results(self, results, aggregation_criteria):
        # Apply aggregation criteria and return aggregated results.
        pass
