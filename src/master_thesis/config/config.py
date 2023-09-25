import os
import logging

logger = logging.getLogger("ConfigManager")


class ConfigManager:
    """Class to manage configuration settings."""

    def __init__(self, environment='development'):
        self.file_path = None
        self.debug = None
        self.batch_size = None
        self.relative_path = None
        self.input_shape = None
        self.show_in_web = None
        self.epochs = None
        self.dataset_name = None
        self.environment = environment
        try:
            self.setup_environment()
        except Exception as e:
            logger.error(f"Error occurred while setting up the environment: {str(e)}", exc_info=True)

    def setup_environment(self):
        """Setup configuration parameters based on the environment."""
        # Common Configuration
        self.dataset_name = "HAM10000"
        self.epochs = 10
        self.show_in_web = True
        self.input_shape = (256, 256, 3)
        self.relative_path = os.path.join('data', 'master-thesis-data')
        self.batch_size = 32  # Example of adding more configuration parameters

        # Environment Specific Configuration
        if self.environment == 'production':
            # Production specific configuration
            self.debug = False
            self.file_path = 'results_epochs_10_prod.json'
        elif self.environment == 'testing':
            # Testing specific configuration
            self.debug = True
            self.file_path = 'results_epochs_10_test.json'
        else:  # development as default
            # Development specific configuration
            self.debug = True
            self.file_path = 'results_epochs_10_dev.json'

    def __str__(self):
        """Return the configuration parameters as a string."""
        parameters = [f"{param} = {value}" for param, value in self.__dict__.items()]
        return "\n".join(parameters)
