import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object
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
        

    def initiate_data_transformation(self, train_path, test_path):
        """
        Function to orchestrate data transformation pipeline
        """
        try:
            target_column_name = 'Perceptions of Electoral Integrity Index'

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Read train and test data completed")

            train_df = train_df.dropna(subset=[target_column_name])
            test_df = test_df.dropna(subset=[target_column_name])

            train_df = self._encode_likert(train_df)
            test_df = self._encode_likert(test_df)
            logging.info("Likert columns encoded")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_obj()

            feature_columns = [
                'State Electoral Integrity', 'National Electoral Integrity', 'Political Scale',
                'Election Rigged_encoded', 'Deadlines Restrictive_encoded', 'Voters Waited_encoded',
                'Voters Intimidated_encoded', 'Multiple Ballots Cast_encoded', 'Machines Accurate_encoded',
                'Records Secure_encoded', 'Votes Counted Quickly_encoded', 'Outcome Reflected Popular Will_encoded'   
            ]

            input_feature_train_df = train_df[feature_columns]
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df[feature_columns]
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        

        

    
