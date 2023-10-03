import os
import time
from typing import Optional, List, Dict

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC, MeanIoU

from src.master_thesis.models.evaluation.metrics import F1Score
from abc import ABC, abstractmethod


class ModelHandlerInterface(ABC):
    @abstractmethod
    def _initialize_model(self):
        pass

    @abstractmethod
    def train(self, train_dataset, validation_dataset, dataset_name, model_name):
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


import logging

logger = logging.getLogger("ModelHandler")


class ModelHandler(ModelHandlerInterface):
    DEFAULT_CLASS_WEIGHT = {0: 1, 1: 1}
    DEFAULT_METRICS = [BinaryAccuracy(), Precision(), Recall(), AUC(), F1Score(), MeanIoU(num_classes=2)]

    def __init__(self, epochs: int = 15, batch_size: int = 32, patience: int = 10,
                 input_shape: Optional[tuple] = None, metrics: Optional[List] = None,
                 scheduler: str = 'constant', save_model: bool = True, class_weight: Optional[Dict[int, int]] = None):
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
        self.logger = logger
        self.logger.info("ModelHandler object has been initialized.")

    @abstractmethod
    def _initialize_model(self):
        pass

    def _validate_train_data(self, model, train_dataset, validation_dataset):
        if not model or not train_dataset or not validation_dataset:
            self.logger.error("Invalid training or validation dataset provided.")
            raise ValueError("Model, Training and Validation dataset cannot be None.")

    def train(self, train_dataset, validation_dataset, dataset_name, model_name):
        self._initialize_model()

        self._validate_train_data(self.model, train_dataset, validation_dataset)

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

        self.logger.info(
            f"Training has been completed in {self.training_time} seconds for {model_name} on {dataset_name}.")
        # Clear GPU memory
        tf.keras.backend.clear_session()

        return self.history, self.training_time

    def evaluate(self, test_dataset):
        t0 = time.perf_counter()
        evaluation_metrics = self.model.evaluate(test_dataset)
        t1 = time.perf_counter()
        self.inference_time = t1 - t0
        self.n_params = self.model.count_params()

        # Clearing the model after evaluation
        tf.keras.backend.clear_session()

        # construct results dict
        results = {
            'train_time': self.training_time,
            'eval_time': self.inference_time,
            'n_params': self.n_params,
            'evaluation_metrics': evaluation_metrics,
            'history': self.history.history
        }
        return results

    def predict(self, x):
        return self.model.predict(x)
