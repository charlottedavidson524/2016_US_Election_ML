import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTranformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def _encode_likert(self, df):
        """
        Function to perform Likert encoding on a copy of the dataset
        """
        likert_map = {
            'Strongly Disagree': 1,
            'Disagree': 2,
            'Neither Agree Nor Disagree': 3,
            'Agree': 4,
            'Strongly Agree': 5,
            "Don't Know": np.nan,
            'Not Applicable': np.nan          
        }

        likert_cols = [
            'Election Rigged', 'Deadlines Restrictive', 'Voters Waited',
            'Voters Intimidated', 'Multiple Ballots Cast', 'Machines Accurate',
            'Records Secure', 'Votes Counted Quickly', 'Outcome Reflected Popular Will'
        ]

        df = df.copy()

        for col in likert_cols:
            df[col + '_encoded'] = df[col].map(likert_map)

        return df

    def get_data_transformer_obj(self):
        """
        Function to return sklearn preprocessor
        """
        try:
            numerical_columns = [
                'State Electoral Integrity',
                'National Electoral Integrity',
                'Political Scale',
                'Election Rigged_encoded',
                'Deadlines Restrictive_encoded',
                'Voters Waited_encoded',
                'Voters Intimidated_encoded',
                'Multiple Ballots Cast_encoded',
                'Machines Accurate_encoded',
                'Records Secure_encoded',
                'Votes Counted Quickly_encoded',
                'Outcome Reflected Popular Will_encoded'
                ]
            
            pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('pipeline', pipeline, numerical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            return CustomException(e, sys)
        

    
