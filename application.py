from flask import Flask,request,render_template
import os
import sys

import numpy as np 
import pandas as pd
#import dill
import pickle


#from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

#app=application

## Route for a home page

@application.route('/')
def index():
    return render_template('index.html') 

@application.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            HighBP=float(request.form.get('HighBP')),
            HighChol=float(request.form.get('HighChol')),
            CholCheck=float(request.form.get('CholCheck')),
            BMI=float(request.form.get('BMI')),
            Smoker=float(request.form.get('Smoker')),
            HeartDiseaseorAttack=float(request.form.get('HeartDiseaseorAttack')),
            PhysActivity=float(request.form.get('PhysActivity')),
            Fruits=float(request.form.get('Fruits')),
            Veggies=float(request.form.get('Veggies')),
            HvyAlcoholConsump=float(request.form.get('HvyAlcoholConsump')),
            GenHlth=float(request.form.get('GenHlth')),
            MentHlth=float(request.form.get('MentHlth')),
            PhysHlth=float(request.form.get('PhysHlth')),
            DiffWalk=float(request.form.get('DiffWalk')),
            Sex=float(request.form.get('Sex')),
            Age=float(request.form.get('Age')),
            Education=float(request.form.get('Education')),
            Income=float(request.form.get('Income'))
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results, probabilities =predict_pipeline.predict(pred_df)
        print("after Prediction")
        result_mapping = {
            0: "No Diabetes",
            1: "Diabetes"
        }

        result_text = result_mapping.get(results[0], "Unknown")
        predicted_probability = probabilities[0][int(results[0])] * 100

        result_with_probability = f"{result_text} (Prediction confidence: {predicted_probability:.2f}%)"

        # Pass form values back to template to preserve them
        form_values = {
            'HighBP': request.form.get('HighBP'),
            'HighChol': request.form.get('HighChol'),
            'CholCheck': request.form.get('CholCheck'),
            'BMI': request.form.get('BMI'),
            'Smoker': request.form.get('Smoker'),
            'HeartDiseaseorAttack': request.form.get('HeartDiseaseorAttack'),
            'PhysActivity': request.form.get('PhysActivity'),
            'Fruits': request.form.get('Fruits'),
            'Veggies': request.form.get('Veggies'),
            'HvyAlcoholConsump': request.form.get('HvyAlcoholConsump'),
            'GenHlth': request.form.get('GenHlth'),
            'MentHlth': request.form.get('MentHlth'),
            'PhysHlth': request.form.get('PhysHlth'),
            'DiffWalk': request.form.get('DiffWalk'),
            'Sex': request.form.get('Sex'),
            'Age': request.form.get('Age'),
            'Education': request.form.get('Education'),
            'Income': request.form.get('Income')
        }

        return render_template('home.html', results=result_with_probability, form_values=form_values)

if __name__=="__main__":
    application.run(host="0.0.0.0")       

