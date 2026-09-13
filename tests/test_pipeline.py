import unittest
import pandas as pd
import os
import pickle

class TestTCGAPipeline(unittest.TestCase):
    def test_data_generation(self):
        self.assertTrue(os.path.exists('data/tcga_expression_matrix.csv'))
        df = pd.read_csv('data/tcga_expression_matrix.csv')
        self.assertIn('Target_Clinical_Risk', df.columns)
        self.assertIn('Patient_ID', df.columns)
        self.assertGreater(len(df), 0)

    def test_model_artifacts(self):
        self.assertTrue(os.path.exists('model/model.pkl'))
        self.assertTrue(os.path.exists('model/pca.pkl'))
        self.assertTrue(os.path.exists('model/features.pkl'))
        
        with open('model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
