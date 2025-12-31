import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:

            # Get the absolute path to the project root directory
            # Go up from src/pipeline/ to the project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))

            model_path=os.path.join(project_root,"artifacts","model.pkl")
            preprocessor_path=os.path.join(project_root,'artifacts','preprocessor.pkl')
            
            print(f"Loading model from: {model_path}")
            print(f"Loading preprocessor from: {preprocessor_path}")
            print("Before Loading")
            
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            #data_scaled=preprocessor.transform(features)
            data_scaled= features
            preds=model.predict(data_scaled)
            probabilities = model.predict_proba(data_scaled)
            return preds, probabilities
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        HighBP: float,
        HighChol: float,
        CholCheck: float,
        BMI: float,
        Smoker: float,
        HeartDiseaseorAttack: float,
        PhysActivity: float,
        Fruits: float,
        Veggies: float,
        HvyAlcoholConsump: float,
        GenHlth: float,
        MentHlth: float,
        PhysHlth: float,
        DiffWalk: float,
        Sex: float,
        Age: float,
        Education: float,
        Income: float):

        self.HighBP = HighBP
        self.HighChol = HighChol
        self.CholCheck = CholCheck
        self.BMI = BMI
        self.Smoker = Smoker
        self.HeartDiseaseorAttack = HeartDiseaseorAttack
        self.PhysActivity = PhysActivity
        self.Fruits = Fruits
        self.Veggies = Veggies
        self.HvyAlcoholConsump = HvyAlcoholConsump
        self.GenHlth = GenHlth
        self.MentHlth = MentHlth
        self.PhysHlth = PhysHlth
        self.DiffWalk = DiffWalk
        self.Sex = Sex
        self.Age = Age
        self.Education = Education
        self.Income = Income

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "HighBP": [self.HighBP],
                "HighChol": [self.HighChol],
                "CholCheck": [self.CholCheck],
                "BMI": [self.BMI],
                "Smoker": [self.Smoker],
                "HeartDiseaseorAttack": [self.HeartDiseaseorAttack],
                "PhysActivity": [self.PhysActivity],
                "Fruits": [self.Fruits],
                "Veggies": [self.Veggies],
                "HvyAlcoholConsump": [self.HvyAlcoholConsump],
                "GenHlth": [self.GenHlth],
                "MentHlth": [self.MentHlth],
                "PhysHlth": [self.PhysHlth],
                "DiffWalk": [self.DiffWalk],
                "Sex": [self.Sex],
                "Age": [self.Age],
                "Education": [self.Education],
                "Income": [self.Income]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
