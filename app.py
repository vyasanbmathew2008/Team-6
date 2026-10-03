from pathlib import Path
import os
import pickle

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    genai = None

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
load_dotenv(ROOT / ".env")

AREAS = {
    "Heart disease": "heart",
    "Diabetes": "diabetes",
    "Infectious disease": "infection"
}

st.set_page_config(page_title="Team 6 Health Predictor", page_icon="🩺")
st.title("Team 6 - Health Predictor")
st.write("Machine-learning prediction for three health areas.")
st.warning("Educational use only. This is not a medical diagnosis.")

def load_model(name):
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        st.error("Model not found. Run train.py first.")
        st.stop()
    with open(path, "rb") as file:
        return pickle.load(file)

area = st.selectbox("Health area", list(AREAS))
bundle = load_model(AREAS[area])

values = {}
for column in bundle["columns"]:
    values[column] = st.text_input(column)

if st.button("Predict"):
    row = pd.DataFrame([values])

    for column in row.columns:
        converted = pd.to_numeric(row[column], errors="coerce")
        if converted.notna().all():
            row[column] = converted

    try:
        result = bundle["model"].predict(row)[0]
        st.success(f"Prediction: {result}")

        if hasattr(bundle["model"], "predict_proba"):
            confidence = max(bundle["model"].predict_proba(row)[0])
            st.write(f"Model confidence: {confidence:.1%}")

        key = os.getenv("GEMINI_API_KEY")
        if key and genai:
            client = genai.Client(api_key=key)
            answer = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=(
                    "Explain this health prediction in simple language. "
                    "Do not diagnose the person or prescribe medicine. "
                    f"Health area: {area}. Prediction: {result}"
                )
            )
            st.subheader("Simple explanation")
            st.write(answer.text)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
