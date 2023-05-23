import unittest
from tensorflow.keras.applications import VGG16, ResNet50
from src.models.model_builder import ModelBuilder

class TestModelBuilder(unittest.TestCase):
    def test_get_model(self):
        model = ModelBuilder.get_model('VGG16', (224, 224, 3), 7)
        self.assertIsInstance(model, VGG16)

        model = ModelBuilder.get_model('ResNet', (224, 224, 3), 7)
        self.assertIsInstance(model, ResNet50)

        with self.assertRaises(ValueError):
            ModelBuilder.get_model('NonExistentModel', (224, 224, 3), 7)

if __name__ == '__main__':
    unittest.main()
