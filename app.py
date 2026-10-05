from pathlib import Path
import json
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-2.5-pro",
]


MODELS = {
    "Heart disease": "heart_disease",
    "Diabetes": "diabetes_dataset",
    "Infectious disease / symptoms": "health_dataset",
}


st.set_page_config(
    page_title="Medical AI Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the prediction model...")
def load_model(dataset_label: str):
    dataset_stem = MODELS[dataset_label]
    model_dir = ROOT / "models"
    model_path = model_dir / f"{dataset_stem}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "No pickle model was found. Run train.py first so it creates "
            f"{model_path}"
        )

    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)

    if not isinstance(bundle, dict) or "pipeline" not in bundle or "feature_schema" not in bundle:
        raise ValueError(
            "The pickle file does not contain the expected model bundle. "
            "Please retrain the selected model with train.py."
        )

    return {
        **bundle,
        "model_path": model_path,
        "scores": bundle.get("scores", {}),
    }


def make_input_form(model_info: dict):
    values = {}
    feature_schema = model_info["feature_schema"]
    columns = model_info["columns"]

    if model_info.get("dataset_name") == "health_dataset":
        st.subheader("Enter symptoms")
        st.caption(
            "Select the symptoms that best match the user input. "
            "The health model maps them to a predicted disease."
        )

        symptom_options = sorted({
            option
            for column in columns
            for option in feature_schema.get(column, {}).get("options", [])
            if option
        })

        selected = st.multiselect(
            "Symptoms from the dataset",
            symptom_options,
            max_selections=len(columns),
            placeholder="Choose one or more symptoms",
        )

        typed = st.text_area(
            "Enter symptoms manually",
            placeholder="Example: fever, cough, fatigue",
            help="Separate multiple symptoms with commas.",
        )

        typed_symptoms = [
            item.strip()
            for item in typed.split(",")
            if item.strip()
        ]

        selected = list(dict.fromkeys(selected + typed_symptoms))[:len(columns)]

        for index, column in enumerate(columns):
            values[column] = (
                selected[index]
                if index < len(selected)
                else np.nan
            )

        return values

    st.subheader("Your information")
    st.caption(
        "Enter the information below to receive an AI-assisted health prediction."
    )

    for index, column in enumerate(columns):
        info = feature_schema[column]
        label = str(column)

        if info["kind"] == "numeric":
            minimum = float(info["min"])
            maximum = float(info["max"])
            default = float(info["default"])

            if not np.isfinite(minimum):
                minimum = -1e6

            if not np.isfinite(maximum):
                maximum = 1e6

            if minimum == maximum:
                maximum = minimum + 1.0

            values[column] = st.number_input(
                label,
                min_value=minimum,
                max_value=maximum,
                value=min(max(default, minimum), maximum),
                key=f"field_{index}",
            )

        else:
            options = info.get("options", [""]) or [""]
            values[column] = st.selectbox(
                label,
                options,
                key=f"field_{index}",
            )

    return values


def explain_prediction(
    prediction: dict,
    text_context: str,
    gemini_model: str,
) -> str:

    # Basic fallback if Gemini is unavailable
    fallback = (
        f"The AI model predicted **{prediction['prediction']}** based on "
        "the information provided.\n\n"
        "This prediction does not confirm a medical condition. "
        "A qualified healthcare professional should review the result "
        "and consider the person's symptoms, medical history, and "
        "appropriate clinical tests."
    )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return fallback

    try:
        client = genai.Client(api_key=api_key)

        request = {
            "health_area": prediction["dataset"],
            "prediction": prediction["prediction"],
            "confidence": prediction.get("confidence"),
            "input_values": prediction.get("input_values", {}),
            "additional_information": text_context[:2000],
        }

        prompt = f"""
You are a healthcare information assistant inside an educational AI
health-prediction application.

The machine-learning system has produced a prediction. Your job is to
explain the result clearly and help the user understand what it could mean.

IMPORTANT:
- The prediction is NOT a confirmed diagnosis.
- Never say or imply that the user definitely has the predicted disease.
- Do not prescribe medication.
- Do not provide medication doses.
- Do not invent symptoms or patient information.
- Do not mention datasets, pickle files, algorithms, Random Forest,
  pipelines, backend code, or software implementation.
- Use only the information provided.
- Keep the explanation practical and easy to understand.
- If the confidence is available, explain that confidence is a model
  estimate and does not represent medical certainty.

Return exactly these four sections:

### What the prediction means
Explain in simple language what the predicted condition generally is
and what this prediction means.

### Why this result may have appeared
Briefly explain the important information supplied by the user that
may be relevant to the prediction. Do not claim that a specific factor
caused the prediction.

### What to do next
Give sensible general next steps. Recommend appropriate medical
follow-up when relevant.

### When to seek urgent help
Mention important emergency warning signs relevant to the predicted
condition. If there are no obvious emergency signs, say that clearly.

End with one short sentence explaining that this is an AI prediction
for educational purposes and should not replace professional medical
evaluation.

Here is the prediction information:

{json.dumps(request, default=str, indent=2)}
"""

        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
        )

        if response and response.text:
            return response.text

        return fallback

    except Exception as exc:
        st.warning(
            "Gemini explanation is temporarily unavailable. "
            "Showing the basic prediction explanation instead."
        )
        return fallback


st.title("🩺 Medical AI Predictor")

st.markdown("### Understand your health information with AI")

st.write(
    "Provide the requested information and receive an AI-assisted "
    "prediction with a clear, easy-to-understand explanation."
)

st.info(
    "For educational use only. This tool does not replace a medical "
    "examination, professional advice, or diagnosis."
)


with st.sidebar:
    st.header("Prediction settings")

    selected_dataset = st.selectbox(
        "Health area",
        list(MODELS),
        index=0,
    )

    st.divider()

    configured_model = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite",
    )

    if configured_model not in GEMINI_MODELS:
        configured_model = "gemini-3.5-flash-lite"

    gemini_choice = st.selectbox(
        "Gemini model",
        [
            "Default / configured",
            *GEMINI_MODELS,
            "Custom model ID",
        ],
        index=0,
        help="Choose the Gemini model used for the prediction explanation.",
    )

    if gemini_choice == "Custom model ID":
        gemini_model = st.text_input(
            "Custom Gemini model ID",
            value=configured_model,
            placeholder="Example: gemini-2.5-flash",
            help="Enter a model ID supported by your Gemini API account.",
        ).strip() or configured_model

    elif gemini_choice == "Default / configured":
        gemini_model = configured_model

    else:
        gemini_model = gemini_choice


try:
    model_info = load_model(selected_dataset)

except Exception as exc:
    st.error(str(exc))
    st.stop()


with st.form("prediction_form"):

    record = make_input_form(model_info)

    text_context = st.text_area(
        "Additional information (optional)",
        placeholder=(
            "Add any non-identifying information you would like "
            "the AI to consider for the explanation."
        ),
    )

    submitted = st.form_submit_button(
        "🔍 Check prediction",
        use_container_width=True,
    )


if submitted:

    row = pd.DataFrame(
        [record],
        columns=model_info["columns"],
    )

    pipeline = model_info["pipeline"]

    label = pipeline.predict(row)[0]

    result = {
        "dataset": selected_dataset,
        "prediction": str(label),
        "model": model_info["model_name"],
        "target": model_info["target"],
        "model_file": model_info["model_path"].name,
        "input_values": {
            str(key): str(value)
            for key, value in record.items()
            if pd.notna(value)
        },
    }

    if model_info.get("dataset_name") == "health_dataset":
        result["input_symptoms"] = [
            str(value)
            for value in record.values()
            if pd.notna(value) and str(value).strip()
        ]

    if hasattr(pipeline, "predict_proba"):

        probabilities = pipeline.predict_proba(row)[0]

        result["confidence"] = round(
            float(np.max(probabilities)),
            4,
        )

        result["class_probabilities"] = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(
                pipeline.classes_,
                probabilities,
            )
        }

    st.divider()

    st.markdown("## 🧪 Your result")

    result_col1, result_col2 = st.columns([2.5, 1])

    with result_col1:
        st.success(f"### {result['prediction']}")

        st.caption(
            "This is an AI-generated prediction based on the "
            "information provided."
        )

    with result_col2:

        confidence = result.get("confidence")

        if confidence is not None:
            st.metric(
                "Prediction confidence",
                f"{confidence:.1%}",
            )

    if result.get("class_probabilities"):

        st.markdown("#### Prediction overview")

        probability_cols = st.columns(
            len(result["class_probabilities"])
        )

        for index, (class_name, probability) in enumerate(
            result["class_probabilities"].items()
        ):

            with probability_cols[index]:
                st.metric(
                    class_name,
                    f"{probability:.1%}",
                )

    st.markdown("### 🤖 What this result means")

    explanation = explain_prediction(
        result,
        text_context,
        gemini_model,
    )

    st.info(explanation)

    st.caption(
        "Important: An AI prediction can be incorrect. "
        "If you have symptoms or health concerns, consult a "
        "qualified healthcare professional."
    )


st.divider()
