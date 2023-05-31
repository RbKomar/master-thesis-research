import unittest
from tensorflow.keras.applications import VGG16, ResNet50
from _old.model_builder import ModelBuilder


class TestModelBuilder(unittest.TestCase):
    def test_get_model(self):
        model = ModelBuilder.get_model('baseline', (224, 224, 3), 7)
        self.assertIsNotNone(model)

        model = ModelBuilder.get_model('VGG16', (224, 224, 3), 7)
        self.assertIsInstance(model, VGG16)

        model = ModelBuilder.get_model('ResNet', (224, 224, 3), 7)
        self.assertIsInstance(model, ResNet50)

        with self.assertRaises(ValueError):
            ModelBuilder.get_model('NonExistentModel', (224, 224, 3), 7)

    def test_get_model_invalid_model_name(self):
        with self.assertRaises(ValueError):
            ModelBuilder.get_model('invalid_model_name', (224, 224, 3), 7)


if __name__ == '__main__':
    unittest.main()
