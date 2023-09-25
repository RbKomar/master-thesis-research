
import logging
from typing import List, Type, Union

from src.master_thesis.config.config import ConfigManager
from src.master_thesis.models.evaluation.evaluator import ModelEvaluator
from src.master_thesis.models.train_model import ModelHandler
from src.master_thesis.models.visualization.visualizer import ModelVisualizer  # Importing ModelVisualizer

logger = logging.getLogger("PipelineController")


class PipelineController:
    def __init__(self,
                 config_manager: ConfigManager,
                 model_evaluator: ModelEvaluator,
                 model_visualizer: ModelVisualizer):  # Specifying type of model_visualizer
        self.config_manager = config_manager
        self.model_evaluator = model_evaluator
        self.model_visualizer = model_visualizer
        self.models_with_imagenet = []
        self.models_without_imagenet = []
        self.results = {}

    def add_models(self, model_trainer: Union[ModelHandler, List[Type[ModelHandler]]]):
        try:
            if isinstance(model_trainer, list):
                for model in model_trainer:
                    self.create_models(model)
            else:
                self.create_models(model_trainer)
        except Exception as e:
            logger.error(f"Error occurred while adding models: {str(e)}", exc_info=True)

    def create_models(self, model_trainer: ModelHandler):
        try:
            model_trainer_with_imagenet = model_trainer(use_imagenet=True,
                                                        epochs=self.config_manager.epochs,
                                                        input_shape=self.config_manager.input_shape)
            model_trainer_without_imagenet = model_trainer(use_imagenet=False,
                                                           epochs=self.config_manager.epochs,
                                                           input_shape=self.config_manager.input_shape)
            self.models_with_imagenet.append(model_trainer_with_imagenet)
            self.models_without_imagenet.append(model_trainer_without_imagenet)
        except Exception as e:
            logger.error(f"Error occurred while creating models: {str(e)}", exc_info=True)

    def run_pipeline(self, use_imagenet=True):
        try:
            models = self.models_with_imagenet if use_imagenet else self.models_without_imagenet

            for model_trainer in models:
                model_trainer.initialize_model()
                model_trainer.train_model()
                evaluation_results = self.model_evaluator.evaluate_model(model_trainer.model)
                self.results[model_trainer.__class__.__name__] = evaluation_results
                self.model_visualizer.plot_training_history(model_trainer.__class__.__name__, model_trainer.history)

            comparison_results = self.model_evaluator.compare_models(self.results)
            self.model_visualizer.plot_model_comparison(comparison_results)
        except Exception as e:
            logger.error(f"Error occurred while running the pipeline: {str(e)}", exc_info=True)
