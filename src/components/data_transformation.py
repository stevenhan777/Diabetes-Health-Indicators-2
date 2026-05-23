import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline  # Use imblearn's Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
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
        try:                     
            drop_cols = ColumnTransformer(
                [
                    ('drop_cols', 'drop', ['NoDocbcCost', 'Stroke', 'AnyHealthcare', 'Sex', 'DiffWalk', 'Smoker', 'Veggies', 'Fruits', 'PhysActivity']) # Explicitly drop these features
                ],
                remainder='passthrough'
            ).set_output(transform="pandas")
            logging.info("Drop cols preprocessor defined")

            # Resampling pipeline — last step IS a resampler, so fit_resample works
            resampling_pipeline = Pipeline(
                steps=[
                    ('drop_cols', drop_cols),
                    ('smote', SMOTENC(
                        categorical_features=['remainder__HighBP','remainder__HighChol','remainder__CholCheck',
                                            'remainder__HeartDiseaseorAttack','remainder__HvyAlcoholConsump'],
                        sampling_strategy={1: 100000},
                        random_state=42)),
                    ('randomundersampler', RandomUnderSampler(
                        sampling_strategy={0: 100000},
                        random_state=42))
                ]
            )

            # Separate scaler applied after resampling
            scaler = MinMaxScaler()


            logging.info("Pipeline created with column dropping, MinMaxScaler, SMOTE, and RandomUnderSampler")

            test_preprocessor = ColumnTransformer(
                [
                    ('drop_cols', 'drop', ['NoDocbcCost', 'Stroke', 'AnyHealthcare', 'Sex', 'DiffWalk', 'Smoker', 'Veggies', 'Fruits', 'PhysActivity']) # Explicitly drop these features
                ],
                remainder='passthrough'
            ).set_output(transform="pandas")
            logging.info("Test Drop cols preprocessor defined")

            # After fitting the full pipeline, extract just the steps needed for inference
            inference_preprocessor = Pipeline(
                steps=[
                    ('drop_cols', drop_cols),
                    ('scaler', scaler)  # reuse already-fitted scaler
                ]
            )

            return drop_cols, resampling_pipeline, scaler, test_preprocessor, inference_preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing and testpreprocessing object")

            drop_cols, resampling_pipeline, scaler_obj,  test_preprocessing_obj, inference_preprocessor = self.get_data_transformer_object()

            target_column_name="Diabetes"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            # output to get the column names
            new_df = drop_cols.fit_transform(input_feature_train_df)
            print("new df cols: ", new_df.columns)

            logging.info("Applying preprocessing object using fit_resample with BOTH features and target on Training")
            logging.info("This applies column dropping, SMOTE, and undersamplingon training dataframe and testing dataframe.")
            X_train_resampled, y_train_resampled  = resampling_pipeline.fit_resample(input_feature_train_df, target_feature_train_df)

            # Step 2: scale after resampling — fit only on resampled training data
            logging.info("Scaling after resampling")
            input_feature_train_arr = scaler_obj.fit_transform(X_train_resampled)

            # Step 3: apply same drop + fitted scaler to test set (no refit)
            logging.info("Applying drop + scale to test set (no refit)")
            X_test_dropped = test_preprocessing_obj.fit(input_feature_train_df).transform(input_feature_test_df)
            input_feature_test_arr = scaler_obj.transform(X_test_dropped)

            train_arr = np.c_[input_feature_train_arr, y_train_resampled]
            test_arr  = np.c_[input_feature_test_arr,  np.array(target_feature_test_df)]

            logging.info("Saving inference preprocessor (drop + scale only, no resampling)")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=inference_preprocessor  # Save this instead of the full pipeline
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)