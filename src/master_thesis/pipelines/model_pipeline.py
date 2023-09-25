from abc import ABC, abstractmethod


# Interfaces

class IModelHandler(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class IEvaluator(ABC):
    @abstractmethod
    def evaluate_model(self, model):
        pass


class IVisualizer(ABC):
    @abstractmethod
    def visualize(self, results):
        pass


# Classes implementing the interfaces

class ModelHandler(IModelHandler):
    def train(self):
        print("Training Model")

    def evaluate(self):
        print("Evaluating Model")
        return {"accuracy": 0.95}


class ModelEvaluator(IEvaluator):
    def evaluate_model(self, model):
        return model.evaluate()


class ModelVisualizer(IVisualizer):
    def visualize(self, results):
        print(f"Visualizing Results: {results}")


class ModelFactory:
    @staticmethod
    def create_model_handler():
        return ModelHandler()


class ModelPipeline:
    def __init__(self, model_handler: IModelHandler, evaluator: IEvaluator, visualizer: IVisualizer):
        self.model_handler = model_handler
        self.evaluator = evaluator
        self.visualizer = visualizer

    def run_pipeline(self):
        self.model_handler.train()
        results = self.evaluator.evaluate_model(self.model_handler)
        self.visualizer.visualize(results)


# Running the pipeline

model_handler = ModelFactory.create_model_handler()
evaluator = ModelEvaluator()
visualizer = ModelVisualizer()

pipeline = ModelPipeline(model_handler, evaluator, visualizer)
pipeline.run_pipeline()
