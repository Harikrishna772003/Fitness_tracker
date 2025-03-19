import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load and prepare data
@st.cache_data
def load_and_prepare_data():
    # Ensure file paths are correct
    calories = pd.read_csv("calories.csv")  
    exercise = pd.read_csv("exercise.csv")

    df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["BMI"] = df["BMI"].round(2)
    df = df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    df = pd.get_dummies(df, drop_first=True)  # Convert categorical columns

    return df

# Train the model
@st.cache_resource
def train_model(df):
    X = df.drop("Calories", axis=1)
    y = df["Calories"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    model.fit(X_train, y_train)
    
    return model, X_train.columns

# Load model and dataset
data = load_and_prepare_data()
model, feature_columns = train_model(data)

# Streamlit UI
st.title("🔥 Personal Fitness Tracker 🔥")
st.write("Enter your details below to estimate calories burned.")

# User inputs
age = st.slider("Age", min_value=10, max_value=100, value=30)
bmi = st.slider("BMI", min_value=15.0, max_value=40.0, value=22.5, step=0.1)
duration = st.slider("Exercise Duration (min)", min_value=0, max_value=35, value=15)
heart_rate = st.slider("Heart Rate (bpm)", min_value=60, max_value=130, value=80)
body_temp = st.slider("Body Temperature (°C)", min_value=36.0, max_value=42.0, value=37.0, step=0.1)
gender = st.radio("Gender", ["Male", "Female"])
gender_value = 1 if gender == "Male" else 0  # Convert to numerical

# Predict button
if st.button("Predict Calories Burned"):
    user_data = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender_value,
    }

    df = pd.DataFrame([user_data])
    df = df.reindex(columns=feature_columns, fill_value=0)  # Ensure correct column order
    prediction = model.predict(df)

    st.success(f"🔥 Estimated Calories Burned: **{round(prediction[0], 2)} kcal** 🔥")
