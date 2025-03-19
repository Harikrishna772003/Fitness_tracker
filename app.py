from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Button, Static, Slider, Select
from textual.containers import Vertical, Horizontal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load and prepare data
def load_and_prepare_data():
    calories = pd.read_csv(""C:\Fitness Tracker\Project3_files\Implementation of Personal Fitness Tracker using Python\calories.csv"")
    exercise = pd.read_csv(""C:\Fitness Tracker\Project3_files\Implementation of Personal Fitness Tracker using Python\exercise.csv"")

    df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["BMI"] = df["BMI"].round(2)
    df = df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    df = pd.get_dummies(df, drop_first=True)

    return df

# Train the model
def train_model(df):
    X = df.drop("Calories", axis=1)
    y = df["Calories"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    model.fit(X_train, y_train)
    return model, X_train.columns

# Load model
data = load_and_prepare_data()
model, feature_columns = train_model(data)

class FitnessApp(App):
    """A Textual Terminal UI for a Fitness Tracker"""

    def compose(self) -> ComposeResult:
        """Define the UI layout"""
        yield Header()
        yield Static("🔥 **Personal Fitness Tracker** 🔥", classes="title")
        
        with Vertical():
            yield Static("Enter your details:")
            yield Slider(name="age", label="Age", min=10, max=100, step=1, value=30)
            yield Slider(name="bmi", label="BMI", min=15, max=40, step=0.1, value=22.5)
            yield Slider(name="duration", label="Duration (min)", min=0, max=35, step=1, value=15)
            yield Slider(name="heart_rate", label="Heart Rate", min=60, max=130, step=1, value=80)
            yield Slider(name="body_temp", label="Body Temp (°C)", min=36, max=42, step=0.1, value=37.0)
            yield Select([("Male", "1"), ("Female", "0")], name="gender", label="Gender")

        with Horizontal():
            yield Button("Predict Calories", name="predict", variant="primary")
            yield Static("", name="result", classes="result")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press to predict calories"""
        if event.button.name == "predict":
            age = self.query_one("#age", Slider).value
            bmi = self.query_one("#bmi", Slider).value
            duration = self.query_one("#duration", Slider).value
            heart_rate = self.query_one("#heart_rate", Slider).value
            body_temp = self.query_one("#body_temp", Slider).value
            gender = int(self.query_one("#gender", Select).value)

            user_data = {
                "Age": age,
                "BMI": bmi,
                "Duration": duration,
                "Heart_Rate": heart_rate,
                "Body_Temp": body_temp,
                "Gender_male": gender,
            }

            df = pd.DataFrame([user_data])
            df = df.reindex(columns=feature_columns, fill_value=0)
            prediction = model.predict(df)

            result_text = f"🔥 Estimated Calories Burned: **{round(prediction[0], 2)} kcal** 🔥"
            self.query_one("#result", Static).update(result_text)

if __name__ == "__main__":
    app = FitnessApp()
    app.run()
