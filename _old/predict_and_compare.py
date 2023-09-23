import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.ai_part.models import ModelBuilder
from src.ai_part.data import load_data

def predict_and_compare(model_name: str, dataset_name: str):
    # Load model
    model = ModelBuilder.get_model(model_name)
    model.load_weights(f'models/{model_name}_{dataset_name}.h5')

    # Load dataset
    dataset = load_data(dataset_name)

    # Predict labels
    predicted_labels = model.predict(dataset.test_images)

    # Compute metrics
    accuracy = accuracy_score(dataset.test_labels, predicted_labels)
    precision = precision_score(dataset.test_labels, predicted_labels, average='weighted')
    recall = recall_score(dataset.test_labels, predicted_labels, average='weighted')
    f1 = f1_score(dataset.test_labels, predicted_labels, average='weighted')

    # Write metrics to a CSV file
    with open(f'results/{model_name}_{dataset_name}_metrics.csv', 'w') as f:
        f.write('accuracy,precision,recall,f1\n')
        f.write(f'{accuracy},{precision},{recall},{f1}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Name of the model to make predictions with")
    parser.add_argument("dataset_name", help="Name of the dataset to predict on")
    args = parser.parse_args()

    predict_and_compare(args.model_name, args.dataset_name)
