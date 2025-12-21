import os
import sys
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(), 
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "GradientBoosting Classifier": GradientBoostingClassifier()  
            }

            params = {
                "Decision Tree": {
                    'max_depth': [3, 5, 7, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2', None],
                    'class_weight': ['balanced', None]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'class_weight': ['balanced', 'balanced_subsample', None],
                    'bootstrap': [True, False]
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.5, 1],
                    'min_child_weight': [1, 3, 5],
                    'scale_pos_weight': [1, 2, 3]                   
                },
                "CatBoost": {
                    'iterations': [50, 100, 200, 300],
                    'depth': [4, 6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'l2_leaf_reg': [1, 3, 5, 7],
                    'border_count': [32, 64, 128],
                    'class_weights': [[1, 1], [1, 2], [1, 3]]                  
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                    'algorithm': ['SAMME', 'SAMME.R']
                },
                "GradientBoosting": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'loss': ['log_loss', 'exponential'],
                }
            }

            model_report = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                            models=models,param=params)

            # Find the best overall model
            print(f"\n\n{'='*60}")
            print("SUMMARY OF ALL MODELS")
            print(f"{'='*60}")

            best_model_score = 0
            best_model_name = None
            best_model_params = None
            best_model_obj = None

            for model_name, results in model_report.items():
                roc_auc = results['test_roc_auc']
                print(f"{model_name}: ROC-AUC: {roc_auc:.4f}")
    
                if roc_auc > best_model_score:
                    best_model_score = roc_auc
                    best_model_name = model_name
                    best_model_params = results['best_params']
                    best_model_obj = results['best_model']

            # Check if best model meets threshold
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            print("BEST OVERALL MODEL")
            print(f"Model: {best_model_name}")
            print(f"ROC-AUC Score: {best_model_score:.4f}")
            print(f"Best Hyperparameters:")
            for param_name, param_value in best_model_params.items():
                print(f"- {param_name}: {param_value}")

            logging.info(f"Best model found: {best_model_name} with ROC-AUC: {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_obj
            )

            logging.info(f"Best model saved successfully")

            # Return best model details
            return {
                'best_model_name': best_model_name,
                'best_roc_auc': best_model_score,
                'best_params': best_model_params
            }
            
        except Exception as e:
            raise CustomException(e,sys)
