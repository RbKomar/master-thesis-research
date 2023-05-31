# src/models/model_comparator.py
import io
import json

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import matplotlib.pyplot as plt


def moving_average(x, w=5):
    return np.convolve(x, np.ones(w), 'valid') / w

SHOW_IN_WEB = True
class ModelComparator:
    def __init__(self):
        self.models_results = {}

    def load_results_from_json(self, file_path):
        with open(file_path, 'r') as file:
            self.models_results = json.load(file)

    def add_model_results(self, model_name, model_trainer):
        self.models_results[model_name] = {
            'history': model_trainer.history,
            'train_time': model_trainer.train_time,
            'inference_time': model_trainer.inference_time,
            'predictions': model_trainer.predictions
        }


    def plot_accuracy(self, dataset_name):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for model_name, results in self.models_results.items():
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(results[dataset_name]['history']['binary_accuracy'])),
                    y=moving_average(results[dataset_name]['history']['binary_accuracy']),
                    mode='lines',
                    name=f'{model_name.split("_")[0]} train'
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(results[dataset_name]['history']['val_binary_accuracy'])),
                    y=moving_average(results[dataset_name]['history']['val_binary_accuracy']),
                    mode='lines',
                    line=dict(dash='dot'),
                    name=f'{model_name.split("_")[0]} val'
                ),
                secondary_y=False,
            )
        fig.update_layout(
            title=f'Model Accuracy ({dataset_name})',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            legend=dict(
                x=0.95,
                y=0.99,
                traceorder="normal",
                font=dict(size=8),
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='rgba(0, 0, 0, 0.5)',
                borderwidth=0.5,
            ),
            yaxis=dict(
                range=[0, 1],
            ),
        )
        fig.update_traces(opacity=0.8)

        if not SHOW_IN_WEB:
            # Render the figure as a PNG image
            img_bytes = fig.to_image(format='png')

            # Display the PNG image using matplotlib
            img = io.BytesIO(img_bytes)
            plt.imshow(plt.imread(img))
            plt.axis('off')
            plt.show()
        else:
            fig.show()

    def plot_auc(self, dataset_name):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for model_name, results in self.models_results.items():
            auc_key = [key for key in results[dataset_name]["history"].keys() if key.startswith('auc')][0]
            val_auc_key = f'val_{auc_key}'
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(results[dataset_name]["history"][auc_key])),
                    y=moving_average(results[dataset_name]["history"][auc_key]),
                    mode='lines',
                    name=f'{model_name.split("_")[0]} val'
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(results[dataset_name]["history"][val_auc_key])),
                    y=moving_average(results[dataset_name]["history"][val_auc_key]),
                    mode='lines',
                    line=dict(dash='dot'),
                    name=f'{model_name.split("_")[0]} val'
                ),
                secondary_y=False,
            )
        fig.update_layout(
            title=f'Model AUC ({dataset_name})',
            xaxis_title='Epoch',
            yaxis_title='AUC',
            legend=dict(
                x=0.95,
                y=0.99,
                traceorder="normal",
                font=dict(size=8),
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='rgba(0, 0, 0, 0.5)',
                borderwidth=0.5,
            ),
            yaxis=dict(
                range=[0, 1],
            ),
            xaxis=dict(
                range=[0, 30],
            ),
        )
        fig.update_traces(opacity=0.8)

        if not SHOW_IN_WEB:
            # Render the figure as a PNG image
            img_bytes = fig.to_image(format='png')

            # Display the PNG image using matplotlib
            img = io.BytesIO(img_bytes)
            plt.imshow(plt.imread(img))
            plt.axis('off')
            plt.show()
        else:
            fig.show()

    def plot_loss(self, dataset_name, is_log=False):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for model_name, results in self.models_results.items():
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(results[dataset_name]['history']['loss'])),
                    y=moving_average(results[dataset_name]['history']['loss']),
                    mode='lines',
                    name=f'{model_name.split("_")[0]} val'
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(results[dataset_name]['history']['val_loss'])),
                    y=moving_average(results[dataset_name]['history']['val_loss']),
                    mode='lines',
                    line=dict(dash='dot'),
                    name=f'{model_name.split("_")[0]} val'
                ),
                secondary_y=False,
            )
        fig.update_layout(
            title=f'Model Loss ({dataset_name})',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            legend=dict(
                x=0.95,
                y=0.99,
                traceorder="normal",
                font=dict(size=8),
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='rgba(0, 0, 0, 0.5)',
                borderwidth=0.5,
            ),
        )
        if is_log:
            fig.update_layout(yaxis_type='log')
            fig.update_layout(yaxis_title='Loss (log scale)')

        fig.update_traces(opacity=0.8)

        if not SHOW_IN_WEB:
            # Render the figure as a PNG image
            img_bytes = fig.to_image(format='png')

            # Display the PNG image using matplotlib
            img = io.BytesIO(img_bytes)
            plt.imshow(plt.imread(img))
            plt.axis('off')
            plt.show()
        else:
            fig.show()

    def compare_inference_time(self):
        model_names = list(self.models_results.keys())
        print(self.models_results.keys())
        inference_times = [results['eval_time'] for results in self.models_results.values()]

        plt.bar(model_names, inference_times)
        plt.title('Model Inference Times')
        plt.ylabel('Inference Time (s)')
        plt.xlabel('Model')
        plt.show()

    def compare_train_time(self):
        model_names = list(self.models_results.keys())
        train_times = [results['train_time'] for results in self.models_results.values()]

        plt.bar(model_names, train_times)
        plt.title('Model Training Times')
        plt.ylabel('Training Time (s)')
        plt.xlabel('Model')
        plt.show()


if __name__ == "__main__":
    print("Model Comparator")
    model_comparator = ModelComparator()
    model_comparator.load_results_from_json('results_epochs_10_x.json')
    model_comparator.plot_accuracy("HAM10000")
    model_comparator.plot_auc("HAM10000")
    model_comparator.plot_loss("HAM10000")
    model_comparator.plot_loss("HAM10000", is_log=True)
    model_comparator.compare_inference_time()
    model_comparator.compare_train_time()
