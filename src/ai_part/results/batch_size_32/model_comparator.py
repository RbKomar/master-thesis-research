# src/models/model_comparator.py
import io
import json

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def moving_average(x, w=5):
    return np.convolve(x, np.ones(w), 'valid') / w


SHOW_IN_WEB = True
EPOCHS = 10


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
                    y=results[dataset_name]['history']['binary_accuracy'],
                    mode='lines',
                    name=f'{model_name.split("_")[0]} train'
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(results[dataset_name]['history']['val_binary_accuracy'])),
                    y=results[dataset_name]['history']['val_binary_accuracy'],
                    mode='lines',
                    line=dict(dash='dot'),
                    name=f'{model_name.split("_")[0]} val'
                ),
                secondary_y=False,
            )
        fig.update_layout(
            title=f'Wyniki treningu - Dokładność ({dataset_name})',
            xaxis_title='Epoki',
            yaxis_title='Dokładność',
            legend=dict(
                x=0.95,
                y=0.99,
                traceorder="normal",
                font=dict(size=16),
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='rgba(0, 0, 0, 0.5)',
                borderwidth=0.5,
            ),
            xaxis=dict(
                range=[0, EPOCHS],
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
                    y=results[dataset_name]["history"][val_auc_key],
                    mode='lines',
                    line=dict(dash='dot'),
                    name=f'{model_name.split("_")[0]} val'
                ),
                secondary_y=False,
            )
        fig.update_layout(
            title=f'Wyniki treningu - AUC ({dataset_name})',
            xaxis_title='Epoki',
            yaxis_title='AUC',
            legend=dict(
                x=0.95,
                y=0.99,
                traceorder="normal",
                font=dict(size=16),
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='rgba(0, 0, 0, 0.5)',
                borderwidth=0.5,
            ),
            xaxis=dict(
                range=[0, EPOCHS],
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

    def plot_f1(self, dataset_name):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for model_name, results in self.models_results.items():
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(results[dataset_name]['history']['f1_score'])),
                    y=moving_average(results[dataset_name]['history']['f1_score']),
                    mode='lines',
                    name=f'{model_name.split("_")[0]} val'
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(results[dataset_name]['history']['val_f1_score'])),
                    y=moving_average(results[dataset_name]['history']['val_f1_score']),
                    mode='lines',
                    line=dict(dash='dot'),
                    name=f'{model_name.split("_")[0]} val'
                ),
                secondary_y=False,
            )
        fig.update_layout(
            title=f'Wyniki treningu - F1 ({dataset_name})',
            xaxis_title='Epoki',
            yaxis_title='Wynik F1',
            legend=dict(
                x=0.95,
                y=0.99,
                traceorder="normal",
                font=dict(size=16),
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='rgba(0, 0, 0, 0.5)',
                borderwidth=0.5,
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
            title=f'Wyniki treningu - Strata ({dataset_name})',
            xaxis_title='Epoki',
            yaxis_title='Strata',
            legend=dict(
                x=0.95,
                y=0.99,
                traceorder="normal",
                font=dict(size=16),
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='rgba(0, 0, 0, 0.5)',
                borderwidth=0.5,
            ),
        )
        if is_log:
            fig.update_layout(yaxis_type='log')
            fig.update_layout(yaxis_title='Strata (skala logarytmiczna)')

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

    def compare_inference_time(self, dataset_name):
        model_names = list(self.models_results.keys())
        print(self.models_results.keys())
        inference_times = [results['eval_time'] for results in self.models_results[dataset_name].values()]

        plt.bar(model_names, inference_times)
        plt.title('Wyniki treningu - czas inferencji')
        plt.ylabel('Czas Inferencji (s)')
        plt.xlabel('Model')
        plt.show()

    def compare_train_time(self, dataset_name):
        model_names = list(self.models_results.keys())
        train_times = [results['train_time'] for results in self.models_results[dataset_name].values()]

        plt.bar(model_names, train_times)
        plt.title('Wyniki treningu - Czas treningu')
        plt.ylabel('Czas Inferencji (s)')
        plt.xlabel('Model')
        plt.show()

    def compare_evaluation_results(self, dataset_name):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib.ticker as mticker
        print([results[dataset_name]['evaluation_metrics'] for results in
               self.models_results.values()])
        model_names = list(self.models_results.keys())
        model_names = [model_name.split("_")[0] for model_name in model_names]
        # evaluation_results for position 0 and 4
        evaluation_results_0 = [results[dataset_name]['evaluation_metrics'][1] for results in
                                self.models_results.values()]
        evaluation_results_4 = [results[dataset_name]['evaluation_metrics'][4] for results in
                                self.models_results.values()]
        print(f"Position 0 results: {evaluation_results_0}")
        print(f"Position 4 results: {evaluation_results_4}")

        # create a figure and a set of subplots
        fig, ax = plt.subplots()

        # bar width
        bar_width = 0.45

        # set the positions of the bars
        index = np.arange(len(model_names))

        # use Seaborn styles
        sns.set_theme()

        # create bars for position 0 and 4
        bar1 = ax.bar(index, evaluation_results_0, bar_width, label="Dokładność")
        bar2 = ax.bar(index + bar_width, evaluation_results_4, bar_width, label="F1")

        # add a legend outside of the plot
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

        # add title and labels
        ax.set_title('Wyniki na zbiorze testowym - modele bez wag z imagenet')
        ax.set_ylabel('Wynik')
        ax.set_xlabel('Model')

        # set the model names as x-axis labels and rotate them for better visibility
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        # add the values on top of the bars
        for bar in bar1 + bar2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01, '%.3f' % float(height), ha='center', va='bottom')

        # Format y axis to 3 decimal places
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

        # adjust the layout to make sure everything fits
        fig.tight_layout()

        # show the plot
        plt.show()


def load_results_from_json(file_path):
    with open(file_path, 'r') as file:
        models_results = json.load(file)
    ham10000 = []
    # get results for models for dataset HAM10000 the file is in model->dataset->results
    for model_name, results in models_results.items():
        models_results[model_name] = results['HAM10000']
        ham10000.append(results['HAM10000'])

    # save it as a json file with ham10000 as the root + file path
    with open('ham10000.json', 'w') as file:
        json.dump(ham10000, file)
    return models_results, ham10000


if __name__ == "__main__":
    print("Wyniki treningu - HAM10000")
    model_comparator = ModelComparator()
    model_comparator.load_results_from_json('results_epochs_10.json')
    # model_comparator.plot_accuracy("HAM10000")
    # model_comparator.plot_f1("HAM10000")
    # model_comparator.plot_loss("HAM10000", is_log=True)
    model_comparator.compare_evaluation_results("HAM10000")
