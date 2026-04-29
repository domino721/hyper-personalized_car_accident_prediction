import pandas as pd
import numpy as np
from src.config import COLUMN_MAPPING, DROP_COLUMNS, ACCIDENT_HISTORY_MAP, CAR_AGE_MAP, CAR_TYPE_MAP, INSURANCE_AMOUNT_MAP, MILEAGE_MAP

class DataProcessor:
    def __init__(self):
        self.column_mapping = COLUMN_MAPPING
        self.drop_columns = DROP_COLUMNS

    def load_and_merge_data(self, file_paths):
        """Loads multiple CSV files and merges them."""
        dfs = []
        for path in file_paths:
            df = pd.read_csv(path, encoding="cp949", engine='python')
            # Fix for df1 issues seen in notebook
            if "차종" in df.columns:
                df = df.dropna(subset=["차종"])
            dfs.append(df)
        
        # Ensure all dfs have same columns as df2 (as per notebook logic)
        target_cols = dfs[1].columns if len(dfs) > 1 else dfs[0].columns
        dfs = [df[target_cols] for df in dfs]
        
        merged_df = pd.concat(dfs, ignore_index=True)
        return merged_df

    def clean_data(self, df):
        """Cleans and renames the dataframe."""
        # Type conversion
        if 'YUHO' in df.columns:
            df['YUHO'] = df['YUHO'].astype(str).str.replace(',', '').astype(int)
        
        if 'SAGO' in df.columns:
            df['SAGO'] = df['SAGO'].apply(lambda x: int(float(str(x).replace(',', '').split('.')[0])))
            
        # Rename columns
        df = df.rename(columns=self.column_mapping)
        
        # Drop unnecessary columns
        df = df.drop(columns=[col for col in self.drop_columns if col in df.columns])
        
        return df

    def feature_engineering(self, df):
        """Creates new features like accident rate and status."""
        df['사고율'] = df.apply(lambda row: 0 if row['유효대수'] == 0 else row['사고건수'] / row['유효대수'], axis=1)
        df['사고유무'] = df.apply(lambda row: 0 if row['사고건수'] == 0 else 1, axis=1)
        return df

    def manual_label_encode(self, df):
        """Performs manual label encoding based on project requirements."""
        df_encoded = df.copy()
        
        # Numeric conversions
        df_encoded['연령대'] = df_encoded['연령대'].apply(lambda x: int(str(x).replace(',', '').split('.')[0]))
        df_encoded['성별'] = df_encoded['성별'].apply(lambda x: 1 if int(str(x).replace('.','').split('.')[0]) == 2 else 0)
        
        # Car source: 1 for Domestic/Missing, 0 for Imported
        df_encoded['국산차량여부'] = df_encoded['국산차량여부'].apply(
            lambda x: 1 if pd.isna(x) or str(x) == "nan" else (0 if int(str(x).replace('.','').split('.')[0]) == 2 else 1)
        )
        
        # Mappings from config
        df_encoded['직전3년간사고건수'] = df_encoded['직전3년간사고건수'].astype(str).str.replace('0', 'N')
        df_encoded['직전3년간사고건수'] = df_encoded['직전3년간사고건수'].map(ACCIDENT_HISTORY_MAP).fillna(0).astype(int)
        
        df_encoded['차량경과년수'] = df_encoded['차량경과년수'].map(CAR_AGE_MAP).fillna(0).astype(int)
        df_encoded['차종'] = df_encoded['차종'].map(CAR_TYPE_MAP).fillna(6).astype(int)
        df_encoded['가입경력코드'] = df_encoded['가입경력코드'].apply(lambda x: int(str(x).split('.')[0]))
        df_encoded['차량가입금액'] = df_encoded['차량가입금액'].map(INSURANCE_AMOUNT_MAP).fillna(0).astype(int)
        df_encoded['영상기록장치특약가입'] = df_encoded['영상기록장치특약가입'].apply(lambda x: 1 if str(x) == '가입' else 0)
        
        # Mileage mapping
        df_encoded['마일리지약정거리'] = df_encoded['마일리지약정거리'].astype(str).map(MILEAGE_MAP).fillna(6).astype(int)
        
        return df_encoded
