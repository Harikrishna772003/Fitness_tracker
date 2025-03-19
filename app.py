import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# --- Page Configuration (Must be first Streamlit command) ---
st.set_page_config(
    page_title="🔥 Personal Fitness Tracker",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #FF5733, #C70039);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #28A745 !important;
        color: white !important;
        border-radius: 10px;
        font-size: 18px;
    }
    .stSlider>div>div {
        background-color: #FF5733 !important;
    }
    .stRadio>div {
        background-color: #FFC300;
        border-radius: 10px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load and Prepare Data ---
@st.cache_data
def load_and_prepare_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")

    df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["BMI"] = df["BMI"].round(2)
    df = df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    df = pd.get_dummies(df, drop_first=True)

    return df

# --- Train the Model ---
@st.cache_resource
def train_model(df):
    X = df.drop("Calories", axis=1)
    y = df["Calories"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    model.fit(X_train, y_train)

    return model, X_train.columns

# --- Load Model ---
data = load_and_prepare_data()
model, feature_columns = train_model(data)

# --- Header Section ---
st.markdown("<h1 style='text-align: center; color: white;'>🔥 Personal Fitness Tracker 🔥</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>Estimate Your Calories Burned Based on Your Workout</h4>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- User Input Section ---
st.markdown("### 🏋️ **Enter Your Details Below:**")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("🔹 Age", min_value=10, max_value=100, value=30)
    bmi = st.slider("🔹 BMI", min_value=15.0, max_value=40.0, value=22.5, step=0.1)

with col2:
    duration = st.slider("⏳ Exercise Duration (min)", min_value=0, max_value=35, value=15)
    heart_rate = st.slider("❤️ Heart Rate (bpm)", min_value=60, max_value=130, value=80)

with col3:
    body_temp = st.slider("🌡️ Body Temperature (°C)", min_value=36.0, max_value=42.0, value=37.0, step=0.1)
    gender = st.radio("⚤ Gender", ["Male", "Female"], horizontal=True)

# Convert gender to numerical
gender_value = 1 if gender == "Male" else 0  

# --- Prediction Button ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>🚀 Predict Your Calories Burned</h3>", unsafe_allow_html=True)

if st.button("🔥 Predict Calories Burned", use_container_width=True):
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

    st.markdown(
        f"""
        <div style='background-color: #28A745; padding: 20px; border-radius: 10px; text-align: center;'>
            <h2 style='color: white;'>🔥 Estimated Calories Burned: {round(prediction[0], 2)} kcal 🔥</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.balloons()  # Fun animation when prediction is made!

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>💪 Stay Fit. Stay Strong. Keep Burning Calories! 💪</h4>", unsafe_allow_html=True)
