import logging

# Importing model handlers
from src.master_thesis.models.networks.vgg import VGGModelHandler
from src.master_thesis.models.networks.efficientnet import EfficientNetModelHandler
from src.master_thesis.models.networks.resnet import ResNetModelHandler
from src.master_thesis.models.networks.baseline import BaselineModelHandler
from src.master_thesis.models.networks.mobilenet import MobileNetModelHandler

# Importing core components
from src.master_thesis.models.evaluation.evaluator import ModelEvaluator
from src.master_thesis.models.visualization.visualizer import ModelVisualizer
from src.master_thesis.pipeline import PipelineController
from src.master_thesis.config.config import ConfigManager


def main():
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Initialize ConfigManager
        config_manager = ConfigManager()
        config_manager.setup_environment()

        # Initializing models
        models = [
            BaselineModelHandler,
            VGGModelHandler,
            ResNetModelHandler,
            MobileNetModelHandler,
            EfficientNetModelHandler
        ]

        # Initializing core components
        model_evaluator = ModelEvaluator()
        model_visualizer = ModelVisualizer()
        pipeline_controller = PipelineController(config_manager, model_evaluator, model_visualizer)

        # Adding models to the pipeline and running the pipeline
        pipeline_controller.add_models(models)
        pipeline_controller.run_pipeline()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
