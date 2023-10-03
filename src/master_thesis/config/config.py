import os
import dataclasses
import logging

logger = logging.getLogger("ConfigManager")


@dataclasses.dataclass
class ConfigManager:
    environment: str = 'development'
    data_config: dict = dataclasses.field(default_factory=dict)
    model_config: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        try:
            self.setup_environment()
        except Exception as e:
            logger.error(f"Error occurred while setting up the environment: {str(e)}", exc_info=True)

    def _setup_development_environment(self):
        self.model_config.update({
            'file_path': 'results_epochs_10_dev.json',
            'debug': True,
        })

    def _setup_testing_environment(self):
        self.model_config.update({
            'file_path': 'results_epochs_10_test.json',
            'debug': True,
        })

    def _setup_production_environment(self):
        self.model_config.update({
            'file_path': 'results_epochs_10_prod.json',
            'debug': False,
        })

    def setup_environment(self):
        self.data_config.update({
            'dataset_name': "HAM10000",
            'epochs': 10,
            'show_in_web': True,
            'input_shape': (256, 256, 3),
            'relative_path': os.path.join('data', 'master-thesis-data'),
            'batch_size': 32,
        })

        self.model_config.update({
            'learning_rate': 0.001,
            'epochs': 10,
            'batch_size': 32,
            'optimizer': 'adam',  # 'adam', 'sgd', 'rmsprop', etc.
            'loss_function': 'categorical_crossentropy',
            'regularization': {
                'l1': 0.0,
                'l2': 0.01
            },
            'dropout_rate': 0.5,
            'momentum': 0.9,  # Used if the optimizer is SGD
            'lr_decay': 0,  # Learning rate decay over each update.
            'weight_initializer': 'he_normal'  # Choose from 'he_normal', 'glorot_uniform', etc.
        })

        environment = os.environ.get('ENVIRONMENT', self.environment)
        if environment == 'production':
            self._setup_production_environment()
        elif environment == 'testing':
            self._setup_testing_environment()
        else:
            self._setup_development_environment()