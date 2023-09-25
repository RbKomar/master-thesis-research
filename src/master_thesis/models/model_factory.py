# Import the required model handler classes
from src.master_thesis.models.baseline import BaselineModelHandler
from src.master_thesis.models.efficientnet import EfficientNetModelHandler
from src.master_thesis.models.mobilenet import MobileNetModelHandler
from src.master_thesis.models.resnet import ResNetModelHandler
from src.master_thesis.models.vgg import VGGModelHandler


class ModelFactory:
    """
    A class used to create model handler objects based on a string identifier.
    """
    # Dictionary to map string identifiers to corresponding model handler classes.
    model_handler_mapping = {
        "BaselineModelHandler": BaselineModelHandler,
        "EfficientNetModelHandler": EfficientNetModelHandler,
        "MobileNetModelHandler": MobileNetModelHandler,
        "ResNetModelHandler": ResNetModelHandler,
        "VGGModelHandler": VGGModelHandler
    }

    @classmethod
    def create_model_handler(cls, identifier, *args, **kwargs):
        """
        A method to create a model handler object based on the given string identifier.
        :param identifier: A string representing the model handler class to be instantiated.
        :param args: Positional arguments to be passed to the model handler class's constructor.
        :param kwargs: Keyword arguments to be passed to the model handler class's constructor.
        :return: An instance of the model handler class corresponding to the given identifier.
        """
        # Get the model handler class from the mapping.
        model_handler_class = cls.model_handler_mapping.get(identifier)

        # Raise an error if the identifier does not match any model handler class.
        if model_handler_class is None:
            raise ValueError(
                f"Invalid identifier: {identifier}. Valid identifiers are {list(cls.model_handler_mapping.keys())}.")

        # Create and return an instance of the model handler class.
        return model_handler_class(*args, **kwargs)


# Example of using Model Factory
try:
    model_handler = ModelFactory.create_model_handler("BaselineModelHandler", num_classes=2)
    print(f"Successfully created a {type(model_handler)} object.")
except ValueError as ve:
    print(ve)
