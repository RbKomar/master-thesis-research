import unittest
import pandas as pd
from src.ai_part.models import evaluate_model

class TestEvaluateModel(unittest.TestCase):
    def test_evaluate_model(self):
        y_true = pd.DataFrame({'label': [0, 1, 1, 0, 1, 0]})
        y_pred = pd.DataFrame({'label': [0, 1, 0, 0, 1, 1]})

        y_true.to_csv('test_y_true.csv', index=False)
        y_pred.to_csv('test_y_pred.csv', index=False)

        metrics = evaluate_model('test_model', 'test_y_true.csv', 'test_y_pred.csv')

        self.assertEqual(metrics['accuracy'], 0.6666666666666666)
        self.assertEqual(metrics['precision'], 0.6666666666666666)
        self.assertEqual(metrics['recall'], 0.6666666666666666)
        self.assertEqual(metrics['f1'], 0.6666666666666666)
        self.assertEqual(metrics['roc_auc'], 0.6666666666666667)

    def test_evaluate_model_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            evaluate_model('test_model', 'invalid_file.csv', 'test_y_pred.csv')

    def test_evaluate_model_mismatched_labels(self):
        y_true = pd.DataFrame({'label': [0, 1, 1, 0, 1, 0, 1]})
        y_pred = pd.DataFrame({'label': [0, 1, 0, 0, 1, 1]})

        y_true.to_csv('test_y_true.csv', index=False)
        y_pred.to_csv('test_y_pred.csv', index=False)

        with self.assertRaises(ValueError):
            evaluate_model('test_model', 'test_y_true.csv', 'test_y_pred.csv')

if __name__ == '__main__':
    unittest.main()
