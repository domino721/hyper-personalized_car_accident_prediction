import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        # Create a dummy dataframe for testing
        self.test_df = pd.DataFrame({
            "ZINSRDAVL": [0.0, 40.0],
            "ZIOSEXCD": [1.0, 2.0],
            "ZDPRODSCD": [np.nan, 1.0],
            "NCR": ["B", "C"],
            "ZCARPSGVL": ["신차", "5년이하"],
            "차종": ["기타", "중형"],
            "ZDRVLISCD___T": ["가족및형제자매한정", "누구나(기본)"],
            "ZENTCARCD": [8.0, 8.0],
            "ZCARISDAM": ["5천만원이하", "미가입"],
            "ZIMAGERVL": ["가입", "미가입"],
            "마일리지약정거리": ["15000K", "7000K"],
            "YUHO": ["1", "0"],
            "SAGO": ["0.0", "0"]
        })

    def test_clean_data(self):
        cleaned_df = self.processor.clean_data(self.test_df)
        self.assertIn("연령대", cleaned_df.columns)
        self.assertIn("사고건수", cleaned_df.columns)
        self.assertEqual(cleaned_df["유효대수"].iloc[0], 1)

    def test_feature_engineering(self):
        cleaned_df = self.processor.clean_data(self.test_df)
        fe_df = self.processor.feature_engineering(cleaned_df)
        self.assertIn("사고율", fe_df.columns)
        self.assertIn("사고유무", fe_df.columns)

    def test_manual_label_encode(self):
        cleaned_df = self.processor.clean_data(self.test_df)
        fe_df = self.processor.feature_engineering(cleaned_df)
        encoded_df = self.processor.manual_label_encode(fe_df)
        
        # Check some values
        self.assertEqual(encoded_df["성별"].iloc[0], 0) # 1.0 -> 0 (Female if 1, Male if 2 in notebook logic?)
        # Let's double check notebook logic: yn_orderlabel_X['성별'] = yn_orderlabel_X['성별'].apply(lambda x: 1 if int(str(x).replace('.','').split('.')[0]) == 2 else 0)
        # So 2.0 -> 1, 1.0 -> 0. Correct.
        self.assertEqual(encoded_df["차량경과년수"].iloc[0], 0) # 신차 -> 0

if __name__ == '__main__':
    unittest.main()
