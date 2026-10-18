
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

model = load_model()

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the four attributes to get the model's prediction.")

age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)

sex = st.selectbox("Sex", ["Female", "Male"])

cp = st.selectbox(
    "Chest pain type",
    [
        "1 - Typical angina",
        "2 - Atypical angina",
        "3 - Non-anginal pain",
        "4 - Asymptomatic"
    ]
)

thalach = st.number_input(
    "Maximum heart rate achieved",
    min_value=40,
    max_value=250,
    value=150
)

if st.button("Predict", type="primary"):
    input_data = pd.DataFrame(
        [[age, 1 if sex == "Male" else 0, int(cp.split(" - ")[0]), thalach]],
        columns=["age", "sex", "cp", "thalach"]
    )

    prediction = int(model.predict(input_data)[0])

    st.subheader("Model prediction")

    if prediction == 1:
        st.warning("The model predicts a positive heart disease label.")
    else:
        st.success("The model predicts a negative heart disease label.")

    st.caption(
        "Educational use only. This prediction is not a medical diagnosis."
    )
