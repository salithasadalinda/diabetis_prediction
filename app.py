import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import pandas as pd
import pickle
# First, install streamlit if you haven't already
# !pip install streamlit
 
# First, install streamlit if you haven't already
# !pip install streamlit
 
import streamlit as st
import pandas as pd
import pickle
 
# Load all models and preprocessing tools from the pickle file
with open('all_trained_models.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
 
# Extract models and preprocessors
loaded_log_reg = loaded_data['Logistic Regression']
loaded_rf_clf = loaded_data['Random Forest']
loaded_svc_model = loaded_data['SVC']
loaded_xgb_model = loaded_data['XGBoost']
loaded_gb_model = loaded_data['Gradient Boosting']
loaded_le = loaded_data['Label Encoder (Gender)']
loaded_scaler = loaded_data['Standard Scaler']
 
# Function to make predictions based on selected model
def predict_diabetes(input_data, model):
    # Scale numerical features
    input_data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = loaded_scaler.transform(
        input_data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
    )
   
    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]  # Return the prediction (0 or 1)
 
# Streamlit app layout
st.title('Diabetes Prediction App')
 
# Input form
st.header('Enter patient data:')
with st.form(key='patient_form'):
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=0, max_value=120, step=1)
    hypertension = st.selectbox('Hypertension', [0, 1])
    heart_disease = st.selectbox('Heart Disease', [0, 1])
    bmi = st.number_input('BMI', min_value=0.0, step=0.1)
    HbA1c_level = st.number_input('HbA1c Level', min_value=0.0, step=0.1)
    blood_glucose_level = st.number_input('Blood Glucose Level', min_value=0.0, step=0.1)
   
    # Dropdown selector for smoking history
    smoking_history = st.selectbox('Smoking History',
                                   [ 'never','current', 'ever', 'former', 'not current'])
 
    # Checkbox to select the model
    st.header('Select a model for prediction:')
    log_reg = st.checkbox('Logistic Regression')
    rf_clf = st.checkbox('Random Forest')
    svc_model = st.checkbox('SVC')
    xgb_model = st.checkbox('XGBoost')
    gb_model = st.checkbox('Gradient Boosting')
 
    # Submit button
    submit_button = st.form_submit_button(label='Submit')
 
# After submitting the form
if submit_button:
    # Convert smoking history into one-hot encoded features
    smoking_history_data = {
        'smoking_history_current': 1 if smoking_history == 'current' else 0,
        'smoking_history_ever': 1 if smoking_history == 'ever' else 0,
        'smoking_history_former': 1 if smoking_history == 'former' else 0,
        'smoking_history_never': 1 if smoking_history == 'never' else 0,
        'smoking_history_not current': 1 if smoking_history == 'not current' else 0,
    }
 
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame([{
        'gender': 1 if gender == 'Female' else 0,  # Assuming 'Female' is 1, 'Male' is 0
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level,
        **smoking_history_data
    }])
 
    # Check which model is selected and make predictions
    if log_reg:
        prediction = predict_diabetes(input_data, loaded_log_reg)
        st.write(f"Prediction using Logistic Regression: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
    if rf_clf:
        prediction = predict_diabetes(input_data, loaded_rf_clf)
        st.write(f"Prediction using Random Forest: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
    if svc_model:
        prediction = predict_diabetes(input_data, loaded_svc_model)
        st.write(f"Prediction using SVC: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
    if xgb_model:
        prediction = predict_diabetes(input_data, loaded_xgb_model)
        st.write(f"Prediction using XGBoost: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
    if gb_model:
        prediction = predict_diabetes(input_data, loaded_gb_model)
        st.write(f"Prediction using Gradient Boosting: {'Diabetes' if prediction == 1 else 'No Diabetes'}")