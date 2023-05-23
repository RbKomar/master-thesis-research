import unittest
import pandas as pd
from src.data.make_dataset import load_data, split_data

class TestMakeDataset(unittest.TestCase):
    def test_load_data(self):
        df = load_data('data/raw/ham10000')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue('image' in df.columns)
        self.assertTrue('label' in df.columns)

    def test_split_data(self):
        df = pd.DataFrame({
            'image': ['img1', 'img2', 'img3', 'img4', 'img5', 'img6'],
            'label': ['A', 'A', 'B', 'B', 'C', 'C']
        })
        train_df, val_df, test_df = split_data(df, seed=42)
        self.assertEqual(len(train_df), 4)
        self.assertEqual(len(val_df), 1)
        self.assertEqual(len(test_df), 1)

if __name__ == '__main__':
    unittest.main()
