import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.master_thesis.models.results.model_results import ModelResults

# from src.master_thesis.models.visualization.visualizer import RobustnessTesterGradCAM

LINES_MARKERS = 'lines+markers'


class ModelPlotter:
    """Base class for plotting model results."""

    def __init__(self, model_results: ModelResults):
        self.model_results = model_results

    def plot(self, dataset_name: str):
        raise NotImplementedError("Derived classes should implement this method.")

    @staticmethod
    def configure_plot(fig, title: str, xaxis_title: str, yaxis_title: str):
        fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
        fig.update_traces(opacity=0.8)


class AccuracyPlotter(ModelPlotter):
    """Class for plotting accuracy of model results."""

    def plot(self, dataset_name: str):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        dataset_results = self.model_results.get_results_for_dataset(dataset_name)
        for model_name, results in dataset_results.items():
            if 'history' in results and 'binary_accuracy' in results['history']:
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(results['history']['binary_accuracy'])),
                        y=results['history']['binary_accuracy'],
                        mode=LINES_MARKERS,
                        name=f'{model_name.split("_")[0]} train',
                        hoverinfo='x+y',
                        hovertemplate='Epoch: %{x}<br>Accuracy: %{y:.3f}'
                    ),
                    secondary_y=False,
                )
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(results['history']['val_binary_accuracy'])),
                        y=results['history']['val_binary_accuracy'],
                        mode=LINES_MARKERS,
                        line=dict(dash='dot'),
                        name=f'{model_name.split("_")[0]} val',
                        hoverinfo='x+y',
                        hovertemplate='Epoch: %{x}<br>Val Accuracy: %{y:.3f}'
                    ),
                    secondary_y=False,
                )
        fig.update_layout(title=f'Training Results - Accuracy ({dataset_name})')
        fig.show()


# Define the Plotter classes with plotly
class AUCPlotter(ModelPlotter):
    def plot(self, dataset_name: str):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        dataset_results = self.model_results.get_results_for_dataset(dataset_name)
        if not dataset_results:
            raise ValueError(f"No results found for dataset: {dataset_name}")
        for model_name, results in dataset_results.items():
            if 'history' in results and 'auc' in results['history']:
                fig.add_trace(
                    go.Scatter(x=np.arange(len(results['history']['auc'])), y=results['history']['auc'],
                               mode=LINES_MARKERS, name=f'{model_name} train AUC'),
                    secondary_y=False,
                )
                if 'val_auc' in results['history']:
                    fig.add_trace(
                        go.Scatter(x=np.arange(len(results['history']['val_auc'])), y=results['history']['val_auc'],
                                   mode=LINES_MARKERS, line=dict(dash='dot'), name=f'{model_name} val AUC'),
                        secondary_y=False,
                    )
        self.configure_plot(fig, f'Training Results - AUC Results {dataset_name}', 'Epochs', 'AUC')
        fig.show()


class F1Plotter(ModelPlotter):
    def plot(self, dataset_name: str):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        dataset_results = self.model_results.get_results_for_dataset(dataset_name)
        if not dataset_results:
            raise ValueError(f"No results found for dataset: {dataset_name}")
        for model_name, results in dataset_results.items():
            if 'history' in results and 'f1_score' in results['history']:
                fig.add_trace(
                    go.Scatter(x=np.arange(len(results['history']['f1_score'])), y=results['history']['f1_score'],
                               mode=LINES_MARKERS, name=f'{model_name} train F1 Score'),
                    secondary_y=False,
                )
                if 'val_f1_score' in results['history']:
                    fig.add_trace(
                        go.Scatter(x=np.arange(len(results['history']['val_f1_score'])),
                                   y=results['history']['val_f1_score'],
                                   mode=LINES_MARKERS, line=dict(dash='dot'), name=f'{model_name} val F1 Score'),
                        secondary_y=False,
                    )
        self.configure_plot(fig, f'Training Results - F1 Score {dataset_name}', 'Epochs', 'F1 Score')
        fig.show()


class LossPlotter(ModelPlotter):
    def plot(self, dataset_name: str):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        dataset_results = self.model_results.get_results_for_dataset(dataset_name)
        if not dataset_results:
            raise ValueError(f"No results found for dataset: {dataset_name}")
        for model_name, results in dataset_results.items():
            if 'history' in results and 'loss' in results['history']:
                fig.add_trace(
                    go.Scatter(x=np.arange(len(results['history']['loss'])), y=results['history']['loss'],
                               mode=LINES_MARKERS, name=f'{model_name} train Loss'),
                    secondary_y=False,
                )
                if 'val_loss' in results['history']:
                    fig.add_trace(
                        go.Scatter(x=np.arange(len(results['history']['val_loss'])), y=results['history']['val_loss'],
                                   mode=LINES_MARKERS, line=dict(dash='dot'), name=f'{model_name} val Loss'),
                        secondary_y=False,
                    )
        self.configure_plot(fig, f'Training Results - Loss Results {dataset_name}', 'Epochs', 'Loss')
        fig.show()


# class RobustnessTesterMetricsPlot(RobustnessTesterGradCAM):
#     def plot_performance_metrics(self, metrics):
#         """Plot the performance metrics (accuracy, precision, recall, F1-score) across the three test datasets."""
#         # Assuming metrics is a dictionary of dictionaries with the following structure:
#         # {
#         #    "original": {"accuracy": 0.9, "precision": 0.8, "recall": 0.85, "f1": 0.82},
#         #    "obscured": {"accuracy": 0.85, "precision": 0.78, "recall": 0.8, "f1": 0.79},
#         #    "incomplete": {"accuracy": 0.8, "precision": 0.75, "recall": 0.77, "f1": 0.76}
#         # }
#         PerformanceVisualization.plot_metrics_comparison(metrics)


if __name__ == '__main__':
    from src.master_thesis.models.results.model_results import MockModelResults

    def plotting():
        mock_model_results = MockModelResults()

        # Create instances of the plotter classes
        accuracy_plotter = AccuracyPlotter(mock_model_results)
        auc_plotter = AUCPlotter(mock_model_results)
        f1_plotter = F1Plotter(mock_model_results)
        loss_plotter = LossPlotter(mock_model_results)

        # Plot using the plotter classes with mock data
        accuracy_plotter.plot('dataset1')
        auc_plotter.plot('dataset1')
        f1_plotter.plot('dataset1')
        loss_plotter.plot('dataset1')


    plotting()
