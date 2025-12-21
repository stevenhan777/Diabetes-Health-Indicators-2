import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para=param[model_name]

            rs = RandomizedSearchCV(
                model,
                para,
                n_iter=50,
                cv=5,
                scoring='roc_auc',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )

            rs.fit(X_train,y_train)

            best_model = rs.best_estimator_

            # Calculate ROC-AUC score
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

            test_roc_auc = roc_auc_score(y_test, y_pred_proba)

            # Store results
            report[model_name] = {
                'test_roc_auc': test_roc_auc,
                'best_params': rs.best_params_,
                'best_model': best_model
            }

            # Print results for this model
            print(f"\n{model_name} Results:")
            print(f"ROC-AUC Score: {test_roc_auc:.4f}")
            print(f"Best Hyperparameters:")
            for param_name, param_value in rs.best_params_.items():
                print(f" - {param_name}: {param_value}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    