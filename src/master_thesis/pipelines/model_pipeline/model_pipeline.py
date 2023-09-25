from src.master_thesis.models.model_factory import ModelFactory
from src.master_thesis.pipelines.model_pipeline.interfaces import IModelHandler, IEvaluator, IVisualizer



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
