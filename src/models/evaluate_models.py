import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def evaluate_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    pred_labels = np.argmax(predictions, axis=1)

    metrics = {'accuracy': accuracy_score(test_labels, pred_labels)}
    metrics['precision'] = precision_score(test_labels, pred_labels, average='weighted')
    metrics['recall'] = recall_score(test_labels, pred_labels, average='weighted')
    metrics['f1_score'] = f1_score(test_labels, pred_labels, average='weighted')

    try:
        metrics['auc_roc'] = roc_auc_score(test_labels, predictions, multi_class='ovr')
    except ValueError:
        pass  # Not all models will support ROC AUC scoring

    metrics['confusion_matrix'] = confusion_matrix(test_labels, pred_labels)

    return metrics

def evaluate_models():
    model_names = ['baseline', 'VGG16', 'ResNet']  # Add more models as needed
    dataset_names = ['ham10000', 'isic2016', 'isic2017', 'isic2019', 'isic2020']  # Add more datasets as needed
    metrics = {}

    for model_name in model_names:
        metrics[model_name] = {}

        for dataset_name in dataset_names:
            # Load model
            model = tf.keras.models.load_model(f'models/{model_name}/{model_name}.h5')

            # Load the test data
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(f'data/processed/{dataset_name}/test', target_size=(224, 224), class_mode='categorical')
            test_data, test_labels = next(test_generator)

            # Evaluate the model
            model_metrics = evaluate_model(model, test_data, test_labels)

            metrics[model_name][dataset_name] = model_metrics

    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        data = {
            model_name: [
                metrics[model_name][dataset_name][metric]
                for dataset_name in dataset_names
            ]
            for model_name in model_names
        }
        # Plot comparison chart
        df = pd.DataFrame(data, index=dataset_names)
        df.plot(kind='bar', title=f'Comparison of {metric}')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f'reports/comparison_{metric}.png')

if __name__ == "__main__":
    evaluate_models()
