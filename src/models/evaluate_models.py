import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

def evaluate_models():
    model_names = ['baseline', 'VGG16', 'ResNet']  # Add more models as needed
    dataset_names = ['ham10000', 'isic2016', 'isic2017', 'isic2019', 'isic2020']  # Add more datasets as needed
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    for metric in metrics:
        data = {}

        for model_name in model_names:
            data[model_name] = []
            for dataset_name in dataset_names:
                # Load metrics from CSV file
                df = pd.read_csv(f'results/{model_name}_{dataset_name}_metrics.csv')
                data[model_name].append(df[metric][0])

        # Plot comparison chart
        df = pd.DataFrame(data, index=dataset_names)
        df.plot(kind='bar', title=f'Comparison of {metric}')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f'results/comparison_{metric}.png')

if __name__ == "__main__":
    evaluate_models()

def evaluate_model(model_name, y_true_path, y_pred_path):
    # Load the true and predicted labels
    y_true = pd.read_csv(y_true_path)
    y_pred = pd.read_csv(y_pred_path)

    # Compute the metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)

    # Print the metrics
    print(f'Model: {model_name}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')
    print(f'ROC AUC: {roc_auc}')
    print(f'Confusion Matrix: \n{conf_mat}')

    # Return the metrics
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_mat
    }

if __name__ == '__main__':
    model_names = ['baseline', ...]  # List of model names
    y_true_path = 'data/processed/y_true.csv'
    results = []

    for model_name in model_names:
        y_pred_path = f'reports/{model_name}_predictions.csv'
        result = evaluate_model(model_name, y_true_path, y_pred_path)
        results.append(result)

    # Convert the results to a DataFrame and save it to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('reports/evaluation_results.csv', index=False)
