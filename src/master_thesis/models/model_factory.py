import importlib
from typing import Type

from src.master_thesis.config.config import logger
from src.master_thesis.models.train_model import ModelHandlerInterface


class ModelFactory:
    model_handler_mapping = {
        "BaselineModelHandler": "src.master_thesis.models.networks.baseline.BaselineModelHandler",
        "EfficientNetModelHandler": "src.master_thesis.models.networks.efficientnet.EfficientNetModelHandler",
        "MobileNetModelHandler": "src.master_thesis.models.networks.mobilenet.MobileNetModelHandler",
        "ResNetModelHandler": "src.master_thesis.models.networks.resnet.ResNetModelHandler",
        "VGGModelHandler": "src.master_thesis.models.networks.vgg.VGGModelHandler"
    }

    @classmethod
    def create_model_handler(cls, identifier: str, *args, **kwargs) -> Type[ModelHandlerInterface]:
        model_path = cls.model_handler_mapping.get(identifier)

        if model_path is None:
            valid_identifiers = list(cls.model_handler_mapping.keys())
            logger.error(f"Invalid identifier: {identifier}. Valid identifiers are {valid_identifiers}.")
            raise ValueError(f"Invalid identifier: {identifier}. Valid identifiers are {valid_identifiers}.")

        module_name, class_name = model_path.rsplit(".", 1)
        ModelClass = getattr(importlib.import_module(module_name), class_name)

        return ModelClass(*args, **kwargs)

    @classmethod
    def model_handler_generator(cls, *args, **kwargs):
        for identifier in cls.model_handler_mapping.keys():
            try:
                yield cls.create_model_handler(identifier, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error while creating {identifier}: {str(e)}")


# for model_handler in ModelFactory.model_handler_generator(num_classes=2):
#     # Here, model_handler is an instance of one of the available model handler classes.
#     # You can perform whatever operations you need with this instance, such as training or evaluation.
#     logger.info(f"Successfully created a {type(model_handler)} object.")
#     model_handler.train(...)  # Example operation
#     model_handler.evaluate(...)  # Example operation
