import unittest
import os
from src.ai_part.models.train_model import train_and_save_model
from _old.model_builder import ModelBuilder

class TestTrainModel(unittest.TestCase):
    def test_train_and_save_model(self):
        model = ModelBuilder.get_model('baseline', (224, 224, 3), 7)
        train_and_save_model(model, 'test_model')
        self.assertTrue(os.path.exists('models/test_model.h5'))

    def test_train_and_save_model_invalid_model(self):
        with self.assertRaises(TypeError):
            train_and_save_model("invalid_model", 'test_model')

    def test_train_and_save_model_empty_model_name(self):
        model = ModelBuilder.get_model('baseline', (224, 224, 3), 7)
        with self.assertRaises(ValueError):
            train_and_save_model(model, '')

if __name__ == '__main__':
    unittest.main()
