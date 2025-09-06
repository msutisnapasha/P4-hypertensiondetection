from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Load the trained model
model = joblib.load("hypertension_model_rf_hpt.pkl")

# Initialize FastAPI app
app = FastAPI(title="Hypertension Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # Allow all headers
)

class PredictionInput(BaseModel):
    Age : int
    Salt_Intake : float
    Stress_Score : int
    Sleep_Duration : float
    BMI : float
    BP_History : str
    Medication : str
    Family_History : str
    Exercise_Level : str
    Smoking_Status : str

    # Age : int
    # Salt_Intake : float
    # Stress_Score : int
    # Sleep_Duration : float
    # BMI : float
    # BP_History_Hypertension : int
    # BP_History_Normal : int
    # BP_History_Prehypertension  : int
    # Medication_ACE_Inhibitor : int
    # Medication_Beta_Blocker  : int
    # Medication_Diuretic  : int
    # Medication_Other : int
    # Family_History_No  : int
    # Family_History_Yes  : int
    # Exercise_Level_High : int
    # Exercise_Level_Low  : int
    # Exercise_Level_Moderate : int
    # Smoking_Status_NonSmoker  : int
    # Smoking_Status_Smoker :int

@app.get("/")
def home():
    return {"message": "Hypertension Prediction API is running!"}

@app.post("/predict")
def predict(data: PredictionInput):
    # Convert input to numpy array
    features = create_features(data)
    # Make prediction
    prediction = model.predict(features)
    return {"prediction": prediction.tolist()}


def create_features(data):
    # Create initial dataframe with numeric fields
    features = pd.DataFrame([{
        'Age': data.Age,
        'Salt_Intake': data.Salt_Intake,
        'Stress_Score': data.Stress_Score,
        'Sleep_Duration': data.Sleep_Duration,
        'BMI': data.BMI,
        'BP_History': data.BP_History,
        'Medication': data.Medication,
        'Family_History': data.Family_History,
        'Exercise_Level': data.Exercise_Level,
        'Smoking_Status': data.Smoking_Status
    }])

    # One-hot encode categorical features with fixed categories to ensure column order
    bp_history_cols = ['Hypertension', 'Normal', 'Prehypertension']
    medication_cols = ['ACE Inhibitor', 'Beta Blocker', 'Diuretic', 'Other']
    family_history_cols = ['No', 'Yes']
    exercise_cols = ['High', 'Low', 'Moderate']
    smoking_cols = ['Non-Smoker', 'Smoker']

    features = pd.get_dummies(
        features,
        columns=['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
    )

    # Ensure all expected columns exist (fill missing ones with 0)
    expected_cols = (
        [f'BP_History_{c}' for c in bp_history_cols] +
        [f'Medication_{c}' for c in medication_cols] +
        [f'Family_History_{c}' for c in family_history_cols] +
        [f'Exercise_Level_{c}' for c in exercise_cols] +
        [f'Smoking_Status_{c}' for c in smoking_cols]
    )

    for col in expected_cols:
        if col not in features.columns:
            features[col] = 0

    # Reorder columns
    final_columns = [
        'Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI'
    ] + expected_cols

    features = features[final_columns]

    return features