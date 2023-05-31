# src/models/train_model.py
import os
import time

import tensorflow as tf
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from tensorflow.keras import callbacks
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import backend as K

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

class ModelTrainer:

    def __init__(self, epochs=15, batch_size=32, patience=10, input_shape=None, metrics=None,
                 scheduler='constant'):
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
        if metrics is None:
            self.metrics = [BinaryAccuracy(), Precision(), Recall(), AUC(), F1Score(), MeanIoU(num_classes=2)]
        else:
            self.metrics = [metric() for metric in metrics]
    def train(self, train_dataset, validation_dataset, dataset_name, model_name):
        start_time = time.time()

        initial_learning_rate = 0.001
        decay_steps = 1000
        decay_rate = 0.96

        lr_schedule = ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=self.metrics)

        checkpoint_path = os.path.join('models', 'checkpoints', dataset_name, model_name, 'model.h5')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint_cb = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True)
        early_stopping_cb = callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)
        class_weights = {0: 0.9, 1: 1}

        self.history = self.model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=validation_dataset,
            callbacks=[checkpoint_cb, early_stopping_cb],
            class_weight=class_weights,
            verbose=1)

        end_time = time.time()
        self.training_time = end_time - start_time

        print(f"Model training finished. Total training time: {self.training_time} seconds")
        # Clear GPU memory
        tf.keras.backend.clear_session()

        return self.history, self.training_time

    def evaluate(self, test_dataset):
        start_eval_time = time.time()
        evaluation_metrics = self.model.evaluate(test_dataset)
        end_eval_time = time.time()

        self.inference_time = end_eval_time - start_eval_time
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
