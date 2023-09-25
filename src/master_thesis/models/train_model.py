import os
import time

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC, MeanIoU

from src.master_thesis.models.evaluation.metrics import F1Score

from abc import ABC, abstractmethod


class ModelHandlerInterface(ABC):

    @abstractmethod
    def train(self, model, train_dataset, validation_dataset, dataset_name, model_name):
        """Trains the provided model with the given datasets."""
        pass

    @abstractmethod
    def evaluate(self, test_dataset):
        """Evaluates the model on the test dataset."""
        pass

    @abstractmethod
    def predict(self, x):
        """Predicts the output given an input x."""
        pass


class ModelHandler(ModelHandlerInterface):
    DEFAULT_CLASS_WEIGHT = {0: 1, 1: 1}
    DEFAULT_METRICS = [BinaryAccuracy(), Precision(), Recall(), AUC(), F1Score(), MeanIoU(num_classes=2)]

    def __init__(self, epochs=15, batch_size=32, patience=10, input_shape=None, metrics=None,
                 scheduler='constant', save_model=True, class_weight=None):
        self.history = None
        self.training_time = None
        self.n_params = None
        self.inference_time = None
        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.input_shape = input_shape
        self.scheduler = scheduler
        self.save_model = save_model
        self.class_weight = class_weight if class_weight else self.DEFAULT_CLASS_WEIGHT
        self.metrics = metrics if metrics else self.DEFAULT_METRICS

    def train(self, model, train_dataset, validation_dataset, dataset_name, model_name):
        self.model = model

        # Early stopping callback
        early_stopping_cb = callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)

        # Class weights
        class_weights = self.class_weight

        # Checkpoint callback
        callbacks_list = [early_stopping_cb]
        if self.save_model:
            checkpoint_path = os.path.join('models', 'checkpoints', dataset_name, model_name, 'model.h5')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            checkpoint_cb = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True)
            callbacks_list.append(checkpoint_cb)

        start_time = time.time()
        self.history = self.model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=validation_dataset,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1)

        end_time = time.time()
        self.training_time = end_time - start_time

        print(f"Model training finished. Total training time: {self.training_time} seconds")
        # Clear GPU memory
        tf.keras.backend.clear_session()

        return self.history, self.training_time

    def evaluate(self, test_dataset):
        t0 = time.perf_counter()
        evaluation_metrics = self.model.evaluate(test_dataset)
        t1 = time.perf_counter()
        self.inference_time = t1 - t0
        self.n_params = self.model.count_params()

        history = {key: value.numpy().tolist() if isinstance(value, tf.Tensor) else value for key, value in
                   self.history.history.items()}
        results = {
            'train_time': self.training_time,
            'eval_time': self.inference_time,
            'n_params': self.n_params,
            'evaluation_metrics': evaluation_metrics,
            'history': history
        }

        return results

    def predict(self, x):
        return self.model.predict(x)
