import matplotlib.pyplot as plt
import pandas as pd

class ModelComparator:
    def __init__(self, models_results, metrics):
        self.models_results = models_results
        self.metrics = metrics

    def plot_comparison(self):
        df = pd.DataFrame(self.models_results)

        for metric in self.metrics:
            df.sort_values(by=metric, ascending=False).plot(x='model_name', y=metric, kind='bar', figsize=(10, 5), legend=None)
            plt.title(f'Comparison of {metric}')
            plt.ylabel(metric)
            plt.grid(True)
            plt.show()

    def save_results(self, path):
        df = pd.DataFrame(self.models_results)
        df.to_csv(path, index=False)


