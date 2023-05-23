import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os
from datetime import datetime
from src.models import model_builder
from ..data.utils import load_data
import yaml


def train_model(args):
    with open(args.config_path) as config_file:
        config = yaml.safe_load(config_file)

    x_train, y_train, x_val, y_val, _, _ = load_data(args.dataset_dir)
    model = model_builder.get_model(config['model_name'], config['input_shape'], config['num_classes'])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=config['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # TensorBoard
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(x_train, y_train,
                        batch_size=config['batch_size'],
                        epochs=config['epochs'],
                        validation_data=(x_val, y_val),
                        callbacks=[tensorboard_callback, early_stopping_callback])

    # Save model
    model.save(os.path.join(args.model_dir, args.model_name + '.h5'))
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--dataset_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('--model_name', type=str, help='Name of the model to train')
    parser.add_argument('--model_dir', type=str, help='Directory to save the trained model')
    parser.add_argument('--config_path', type=str, help='Path to configuration file')
    args = parser.parse_args()

    train_model(args)
