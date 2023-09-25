from abc import ABC, abstractmethod


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
