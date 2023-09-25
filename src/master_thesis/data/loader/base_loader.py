from abc import ABC, abstractmethod


class BaseLoader(ABC):

    @abstractmethod
    def load(self, data_path, batch_size=32, obscure_images_percent=0.0):
        pass
