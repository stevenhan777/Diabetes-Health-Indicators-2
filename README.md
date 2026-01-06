# Diabetes-Health-Indicators

### Life cycle of Machine learning Project:

* Understanding the Problem Statement
* Data Collection
* Data Checks to perform
* Exploratory data analysis
* Data Pre-Processing and feature engineering
* Model Training
* Feature selection
* Hyperparameter Tuning
* Best model
* Flask web app
* AWS Deployment

#### 1) Problem Statement

This project aims to understand whether various healthcare and lifestyle factors, such as high blood pressure, high cholesterol, BMI, amount of physical activity, and age, affect a patient's classification as having diabetes or not having diabetes.

#### 2) Data Collection

The data is obtained from the UC Irvine Machine Learning Repository, labeled as CDC Diabetes Health Indicators
https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

#### 3) EDA Notebook

Data cleaning, exploratory data analysis and feature engineering is performed. Highly correlated features and features with low mutual information gain are removed. The dataset is balanced with SMOTE oversample and Random undersampling. 4 datasets are produced for testing:
* unbalanced data with dependent features unscaled
* unbalanced data with dependent features scaled
* balanced data with dependent features unscaled
* balanced data with dependent features scaled

#### 4) Model Training Notebook

The models are fitted on the 4 datasets and determined that the unscaled and balanced dataset has the best overall results and ROC AUC score. All the ensemble boosting models, i.e. XGBoost, AdaBoost, GradientBoost, CatBoost perform the best with CatBoost with the best ROC AUC score. 

#### 5) Feature Selection Notebook

Perform feature selection using the Gradient Boosting model and the unscaled and balanced dataset. Try iterative feature removal that removes 2 features at a time. Then try Recursive feature elimination with cross-validation (RFECV) to select optimal features. End result is removing 2 features with the lowest feature importance score. 

#### 6) Hyperparameter Tuning Notebook

Here, I perform hyperparameter tuning on the most promising models: Decision Tree, Random Forest, XGBoost, CatBoost, AdaBoost, GradientBoost. Final result is the CatBoost model after hyperparameter tuning has the best ROC AUC and PR AUC score. 


#### 7) Modular implementation

I then implemented my process in a modular way within the components folder with a data_ingestion, data_transformation and model_trainer steps. I transform the input data in a pipeline and perform hyperparameter tuning for best result. 

#### 8) Flask web app implementation

I then create a predict_pipeline, application and home.html file to create a simple web application for the user to input the fields and get a prediction of Diabetes or No Diabetes using the best model pickle file. 

#### 9) Deployment on AWS

I then use AWS Elastic Beanstalk and CodePipeline to deploy the Flask web app. The user can successfully predict Diabetes Classification using the best model. Deployment Link: http://diabetesindicatorsclassifier-env.eba-cp4ukfpd.us-east-1.elasticbeanstalk.com/predictdata

#### Conclusion:

This project successfully developed and deployed a machine learning model for diabetes prediction, addressing the challenges of working with imbalanced healthcare data. Through exploratory data analysis, I identified key predictive features and their relationships with diabetes outcomes. Given the clinical implications where both false positives and false negatives carry significant consequences, I prioritized evaluation metrics beyond simply accuracy and took into account precision, recall, and F1-score to ensure model reliability. Succesful deployment necessitated consistent versioning and an intuitive user experience. This project demonstrates a practical framework for healthcare prediction tasks that can contribute to early detection and improved patient outcomes.