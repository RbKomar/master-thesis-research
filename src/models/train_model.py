import argparse
from src.models import ModelBuilder
from src.data import load_data  # Assuming this module contains load_{dataset_name} functions

def train_model(model_name: str, dataset_name: str):
    # Load model from ModelBuilder
    model = ModelBuilder.get_model(model_name)

    # Load dataset
    dataset = load_data(dataset_name)

    # Compile and train the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(dataset.train_images, dataset.train_labels, epochs=10)

    # Save the model
    model.save(f'models/{model_name}_{dataset_name}.h5')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="Name of the model to train")
    parser.add_argument("dataset_name", help="Name of the dataset to train on")
    args = parser.parse_args()

    train_model(args.model_name, args.dataset_name)
