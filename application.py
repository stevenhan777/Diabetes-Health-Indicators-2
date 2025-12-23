from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
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

        return render_template('home.html', results=result_with_probability)

    

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    application.run(host='0.0.0.0', port=port)


# if __name__=="__main__":
#     app.run(host="0.0.0.0")        

