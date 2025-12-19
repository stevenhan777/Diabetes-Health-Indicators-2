import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
#from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline  # Use imblearn's Pipeline
#from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:         
            
            drop_cols = ColumnTransformer(
                [
                    ('drop_cols', 'drop', ['NoDocbcCost', 'Stroke', 'AnyHealthcare']) # Explicitly drop these features
                ],
                remainder='passthrough'
            )
            logging.info("Drop cols preprocessor defined")

            preprocessor = Pipeline(
                steps = [
                    ('drop_cols', drop_cols),
                    ('smote', SMOTE(sampling_strategy={1: 100000}, random_state=42)),
                    ('randomundersampler', RandomUnderSampler(sampling_strategy={0: 100000}, random_state=42))
                ]
            )
            logging.info("Pipeline created with column dropping, SMOTE, and RandomUnderSampler")

            test_preprocessor = ColumnTransformer(
                [
                    ('drop_cols', 'drop', ['NoDocbcCost', 'Stroke', 'AnyHealthcare']) # Explicitly drop these features
                ],
                remainder='passthrough'
            )
            logging.info("Test Drop cols preprocessor defined")

            return preprocessor, test_preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing and testpreprocessing object")

            preprocessing_obj, test_preprocessing_obj = self.get_data_transformer_object()

            target_column_name="Diabetes"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing object using fit_resample with BOTH features and target on Training")
            logging.info("This applies column dropping, SMOTE, and undersamplingon training dataframe and testing dataframe.")
            input_feature_train_arr, target_feature_train_arr = preprocessing_obj.fit_resample(input_feature_train_df, target_feature_train_df)

            logging.info("Applying test preprocessing object using fit_transform with ONLY features on Testing")
            test_preprocessing_obj.fit(input_feature_train_df)
            input_feature_test_arr=test_preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, target_feature_train_arr
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)