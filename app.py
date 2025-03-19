import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# --- Page Configuration ---
st.set_page_config(
    page_title="🔥 Personal Fitness Tracker",
    page_icon="💪",
    layout="wide"
)

# --- Custom CSS for Stylish UI ---
st.markdown("""
    <style>
        body {
            background-color: #1E1E1E;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .stTextInput>div>div>input {
            background-color: #333;
            color: white;
            border-radius: 10px;
            padding: 10px;
        }
        .stRadio>div>label {
            color: white;
            font-weight: bold;
        }
        .stButton>button {
            background-color: #FF5733 !important;
            color: white !important;
            font-size: 20px;
            padding: 10px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #C70039 !important;
        }
        .result-box {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #FFC300;
        }
        .footer {
            text-align: center;
            font-size: 16px;
            color: #FFD700;
            margin-top: 30px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

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

# --- Train Model ---
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

# --- Page Title ---
st.markdown("<h1 style='text-align: center; color: #FFC300;'>🔥 Personal Fitness Tracker 🔥</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #FFD700;'>Enter Your Details to Estimate Calories Burned</h4>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- User Input Form ---
with st.form("user_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input("🎂 Enter Your Age", value="30")
        bmi = st.text_input("⚖️ Enter Your BMI", value="22.5")
        duration = st.text_input("⏳ Enter Exercise Duration (min)", value="15")

    with col2:
        heart_rate = st.text_input("❤️ Enter Your Heart Rate (bpm)", value="80")
        body_temp = st.text_input("🌡️ Enter Body Temperature (°C)", value="37.0")
        gender = st.radio("⚤ Select Gender", ["Male", "Female"], horizontal=True)

    submit_button = st.form_submit_button("🔥 Predict Calories Burned")

# --- Convert Inputs & Predict ---
if submit_button:
    try:
        # Convert input values
        age = int(age)
        bmi = float(bmi)
        duration = int(duration)
        heart_rate = int(heart_rate)
        body_temp = float(body_temp)
        gender_value = 1 if gender == "Male" else 0  # Convert to numerical

        # Prepare input data
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

        st.markdown(f"""
            <div class='result-box'>
                🔥 Estimated Calories Burned: <b>{round(prediction[0], 2)} kcal</b> 🔥
            </div>
        """, unsafe_allow_html=True)

        st.balloons()  # Fun animation on success

    except ValueError:
        st.error("❌ Please enter valid numerical values for all fields.")

# --- Footer (Designed by You) ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'>🔥 App Designed by <b>T.HARIKRISHNA</b> | Stay Fit! 💪</div>", unsafe_allow_html=True)
