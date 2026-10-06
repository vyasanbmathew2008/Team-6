from pathlib import Path
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


# ============================================================
# GEMINI CONFIGURATION
# Supports:
#   1. Local .env
#   2. Streamlit Cloud Secrets
# ============================================================

def get_secret(name: str, default: str = "") -> str:

    try:
        value = st.secrets.get(name)

        if value is not None:
            return str(value).strip()

    except Exception:
        pass

    value = os.getenv(name)

    if value:
        return value.strip()

    return default


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

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


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Medical AI Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GENERAL
    ------------------------------------------------------- */

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* -------------------------------------------------------
       GEMINI HEADER
    ------------------------------------------------------- */

    .ai-header {
        background: linear-gradient(
            135deg,
            rgba(99, 102, 241, 0.12),
            rgba(59, 130, 246, 0.08)
        );
        border: 1px solid rgba(99, 102, 241, 0.20);
        border-radius: 18px;
        padding: 22px 24px;
        margin: 18px 0 20px 0;
    }

    .ai-header-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .ai-header-subtitle {
        font-size: 0.95rem;
        opacity: 0.75;
        line-height: 1.5;
    }

    /* -------------------------------------------------------
       INFORMATION CARDS
    ------------------------------------------------------- */

    .medical-card {
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        border: 1px solid rgba(128, 128, 128, 0.20);
        background: rgba(128, 128, 128, 0.045);
        min-height: 150px;
    }

    .medical-card-title {
        font-size: 1.12rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .medical-card-content {
        font-size: 0.96rem;
        line-height: 1.65;
    }

    .meaning-card {
        border-left: 5px solid #6366f1;
    }

    .why-card {
        border-left: 5px solid #3b82f6;
    }

    .next-card {
        border-left: 5px solid #10b981;
    }

    .urgent-card {
        border-left: 5px solid #ef4444;
    }

    /* -------------------------------------------------------
       AI DISCLAIMER
    ------------------------------------------------------- */

    .ai-disclaimer {
        border-radius: 15px;
        padding: 17px 20px;
        margin-top: 20px;
        border: 1px solid rgba(245, 158, 11, 0.30);
        background: rgba(245, 158, 11, 0.08);
        font-size: 0.88rem;
        line-height: 1.55;
    }

    /* -------------------------------------------------------
       RESULT CARD
    ------------------------------------------------------- */

    .prediction-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.65;
        margin-bottom: 5px;
    }

    .prediction-value {
        font-size: 1.65rem;
        font-weight: 750;
        margin-bottom: 5px;
    }

    /* -------------------------------------------------------
       GEMINI STATUS
    ------------------------------------------------------- */

    .ai-powered {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.20);
        margin-top: 10px;
    }

    /* -------------------------------------------------------
       MOBILE
    ------------------------------------------------------- */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .medical-card {
            padding: 18px;
        }

        .prediction-value {
            font-size: 1.35rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD ML MODEL
# ============================================================

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

    if (
        not isinstance(bundle, dict)
        or "pipeline" not in bundle
        or "feature_schema" not in bundle
    ):
        raise ValueError(
            "The pickle file does not contain the expected model bundle. "
            "Please retrain the selected model with train.py."
        )

    return {
        **bundle,
        "model_path": model_path,
        "scores": bundle.get("scores", {}),
    }


# ============================================================
# INPUT FORM
# ============================================================

def make_input_form(model_info: dict):

    values = {}

    feature_schema = model_info["feature_schema"]
    columns = model_info["columns"]

    # --------------------------------------------------------
    # INFECTIOUS DISEASE
    # --------------------------------------------------------

    if model_info.get("dataset_name") == "health_dataset":

        st.subheader("Enter symptoms")

        st.caption(
            "Select the symptoms that best match the user input. "
            "The health model maps them to a predicted disease."
        )

        symptom_options = sorted({
            option
            for column in columns
            for option in feature_schema.get(column, {}).get(
                "options", []
            )
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

        selected = list(
            dict.fromkeys(selected + typed_symptoms)
        )[:len(columns)]

        for index, column in enumerate(columns):

            values[column] = (
                selected[index]
                if index < len(selected)
                else np.nan
            )

        return values

    # --------------------------------------------------------
    # OTHER HEALTH MODELS
    # --------------------------------------------------------

    st.subheader("Your information")

    st.caption(
        "Enter the information below to receive an AI-assisted "
        "health prediction."
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
                value=min(
                    max(default, minimum),
                    maximum,
                ),
                key=f"field_{index}",
            )

        else:

            options = (
                info.get("options", [""])
                or [""]
            )

            values[column] = st.selectbox(
                label,
                options,
                key=f"field_{index}",
            )

    return values


# ============================================================
# GEMINI EXPLANATION
# ============================================================

def explain_prediction(
    prediction: dict,
    text_context: str,
    gemini_model: str,
) -> str:

    fallback = (
        f"### What the prediction means\n\n"
        f"The AI model predicted **{prediction['prediction']}** "
        "based on the information provided.\n\n"

        "### Why this result may have appeared\n\n"
        "The prediction is based on the information entered into "
        "the application. It does not mean that any particular "
        "factor definitely caused the result.\n\n"

        "### What to do next\n\n"
        "Consider discussing the result and any symptoms or "
        "concerns with a qualified healthcare professional.\n\n"

        "### When to seek urgent help\n\n"
        "If you experience severe, sudden, or rapidly worsening "
        "symptoms, seek urgent medical attention.\n\n"

        "This is an AI prediction for educational purposes and "
        "does not replace professional medical evaluation."
    )

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if not GEMINI_API_KEY:

        st.warning(
            "GEMINI_API_KEY was not found. "
            "Please add it to Streamlit Secrets."
        )

        return fallback

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    try:

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        request = {
            "health_area": prediction["dataset"],
            "prediction": prediction["prediction"],
            "confidence": prediction.get("confidence"),
            "input_values": prediction.get(
                "input_values",
                {},
            ),
            "additional_information": text_context[:2000],
        }

        prompt = f"""
You are a healthcare information assistant inside an
educational AI health-prediction application.

A machine-learning system has produced the prediction below.

Explain the prediction in simple, patient-friendly language.

IMPORTANT RULES:

- The prediction is NOT a confirmed diagnosis.
- Never say the user definitely has the condition.
- Do not prescribe medication.
- Do not provide medication doses.
- Do not invent symptoms or patient information.
- Do not mention datasets, pickle files, algorithms,
  pipelines, backend code, or software implementation.
- Use only the information provided.
- Model confidence is NOT medical certainty.
- Explain that the prediction can be incorrect.

Return exactly these sections:

### What the prediction means

Explain what the predicted condition generally means
and what this prediction indicates.

### Why this result may have appeared

Briefly explain relevant information supplied by the user
that may be associated with the prediction.

Do not claim that a particular factor definitely caused
the prediction.

### What to do next

Give sensible general next steps and recommend appropriate
medical follow-up when relevant.

### When to seek urgent help

Mention important emergency warning signs relevant to the
predicted condition.

If there are no obvious emergency warning signs,
say that clearly.

Finish with a short statement that this is an AI prediction
for educational purposes and does not replace professional
medical evaluation.

Prediction information:

{json.dumps(request, default=str, indent=2)}
"""

        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
        )

        if response is None or not response.text:

            st.error(
                "Gemini returned an empty response."
            )

            return fallback

        return response.text

    except Exception as exc:

        st.error(
            f"Gemini API error: {type(exc).__name__}: {exc}"
        )

        return fallback


# ============================================================
# GEMINI RESPONSE PARSER
# ============================================================

def parse_gemini_response(text: str):

    sections = {
        "What the prediction means": "",
        "Why this result may have appeared": "",
        "What to do next": "",
        "When to seek urgent help": "",
    }

    if not text:
        return sections

    # Normalize headings
    cleaned = text.replace("\r\n", "\n").strip()

    pattern = re.compile(
        r"###\s*(What the prediction means|"
        r"Why this result may have appeared|"
        r"What to do next|"
        r"When to seek urgent help)"
        r"\s*\n?",
        re.IGNORECASE,
    )

    matches = list(pattern.finditer(cleaned))

    if not matches:
        sections["What the prediction means"] = cleaned
        return sections

    for index, match in enumerate(matches):

        heading = match.group(1)

        # Match canonical heading
        canonical = next(
            (
                key
                for key in sections
                if key.lower() == heading.lower()
            ),
            heading,
        )

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(cleaned)

        content = cleaned[start:end].strip()

        sections[canonical] = content

    return sections


# ============================================================
# GEMINI UI RENDERER
# ============================================================

def render_gemini_response(
    explanation: str,
    prediction_label: str,
    gemini_model: str,
):

    sections = parse_gemini_response(
        explanation
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="ai-header">

            <div class="ai-header-title">
                🤖 AI Health Explanation
            </div>

            <div class="ai-header-subtitle">
                A simple explanation of the prediction
                <strong>{prediction_label}</strong>
                based on the information provided.
            </div>

            <div class="ai-powered">
                ✨ Powered by Gemini · {gemini_model}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # WHAT IT MEANS
    # --------------------------------------------------------

    meaning = sections[
        "What the prediction means"
    ]

    if meaning:

        st.markdown(
            f"""
            <div class="medical-card meaning-card">

                <div class="medical-card-title">
                    🧠 What the prediction means
                </div>

                <div class="medical-card-content">
                    {markdown_to_html(meaning)}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # WHY
    # --------------------------------------------------------

    why = sections[
        "Why this result may have appeared"
    ]

    if why:

        st.markdown(
            f"""
            <div class="medical-card why-card">

                <div class="medical-card-title">
                    🔎 Why this result may have appeared
                </div>

                <div class="medical-card-content">
                    {markdown_to_html(why)}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # NEXT STEPS
    # --------------------------------------------------------

    next_steps = sections[
        "What to do next"
    ]

    if next_steps:

        st.markdown(
            f"""
            <div class="medical-card next-card">

                <div class="medical-card-title">
                    ✅ What to do next
                </div>

                <div class="medical-card-content">
                    {markdown_to_html(next_steps)}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # URGENT HELP
    # --------------------------------------------------------

    urgent = sections[
        "When to seek urgent help"
    ]

    if urgent:

        st.markdown(
            f"""
            <div class="medical-card urgent-card">

                <div class="medical-card-title">
                    🚨 When to seek urgent help
                </div>

                <div class="medical-card-content">
                    {markdown_to_html(urgent)}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="ai-disclaimer">

            <strong>⚠️ Important medical disclaimer</strong><br><br>

            This result is an AI-generated prediction for
            educational purposes only. It is not a diagnosis
            and should not be used as a substitute for a
            qualified healthcare professional.

            <br><br>

            AI predictions can be incorrect. If you have
            concerning, severe, sudden, or worsening symptoms,
            seek appropriate medical care.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MARKDOWN → SIMPLE HTML
# ============================================================

def markdown_to_html(text: str) -> str:

    if not text:
        return ""

    # Escape HTML-sensitive characters
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    # Bold
    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"<strong>\1</strong>",
        text,
    )

    # Italic
    text = re.sub(
        r"\*(.*?)\*",
        r"<em>\1</em>",
        text,
    )

    lines = text.split("\n")

    html_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Bullet points
        if line.startswith("- "):

            html_lines.append(
                f"<div style='margin:6px 0 6px 10px;'>"
                f"• {line[2:]}"
                f"</div>"
            )

        elif line.startswith("* "):

            html_lines.append(
                f"<div style='margin:6px 0 6px 10px;'>"
                f"• {line[2:]}"
                f"</div>"
            )

        # Numbered list
        elif re.match(r"^\d+\.\s+", line):

            html_lines.append(
                f"<div style='margin:6px 0 6px 10px;'>"
                f"{line}"
                f"</div>"
            )

        else:

            html_lines.append(
                f"<div style='margin-bottom:9px;'>"
                f"{line}"
                f"</div>"
            )

    return "".join(html_lines)


# ============================================================
# HEADER
# ============================================================

st.title("🩺 Medical AI Predictor")

st.markdown(
    "### Understand your health information with AI"
)

st.write(
    "Provide the requested information and receive an "
    "AI-assisted prediction with a clear, easy-to-understand "
    "explanation."
)

st.info(
    "For educational use only. This tool does not replace "
    "a medical examination, professional advice, or diagnosis."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Prediction settings")

    selected_dataset = st.selectbox(
        "Health area",
        list(MODELS),
        index=0,
    )

    st.divider()

    configured_model = get_secret(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    )

    if configured_model not in GEMINI_MODELS:
        configured_model = "gemini-2.5-flash"

    gemini_choice = st.selectbox(
        "Gemini model",
        [
            "Default / configured",
            *GEMINI_MODELS,
            "Custom model ID",
        ],
        index=0,
    )

    if gemini_choice == "Custom model ID":

        gemini_model = st.text_input(
            "Custom Gemini model ID",
            value=configured_model,
            placeholder="Example: gemini-2.5-flash",
        ).strip() or configured_model

    elif gemini_choice == "Default / configured":

        gemini_model = configured_model

    else:

        gemini_model = gemini_choice


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model_info = load_model(
        selected_dataset
    )

except Exception as exc:

    st.error(str(exc))
    st.stop()


# ============================================================
# FORM
# ============================================================

with st.form("prediction_form"):

    record = make_input_form(
        model_info
    )

    text_context = st.text_area(
        "Additional information (optional)",
        placeholder=(
            "Add any non-identifying information you "
            "would like the AI to consider."
        ),
    )

    submitted = st.form_submit_button(
        "🔍 Check prediction",
        use_container_width=True,
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    try:

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
            "model_file": model_info[
                "model_path"
            ].name,
            "input_values": {
                str(key): str(value)
                for key, value in record.items()
                if pd.notna(value)
            },
        }

        # ----------------------------------------------------
        # SYMPTOMS
        # ----------------------------------------------------

        if (
            model_info.get("dataset_name")
            == "health_dataset"
        ):

            result["input_symptoms"] = [
                str(value)
                for value in record.values()
                if (
                    pd.notna(value)
                    and str(value).strip()
                )
            ]

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if hasattr(
            pipeline,
            "predict_proba",
        ):

            probabilities = pipeline.predict_proba(
                row
            )[0]

            result["confidence"] = round(
                float(np.max(probabilities)),
                4,
            )

            result["class_probabilities"] = {
                str(cls): round(
                    float(prob),
                    4,
                )
                for cls, prob in zip(
                    pipeline.classes_,
                    probabilities,
                )
            }

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            "## 🧪 Your result"
        )

        result_col1, result_col2 = st.columns(
            [2.5, 1]
        )

        with result_col1:

            st.success(
                f"### {result['prediction']}"
            )

            st.caption(
                "This is an AI-generated prediction "
                "based on the information provided."
            )

        with result_col2:

            confidence = result.get(
                "confidence"
            )

            if confidence is not None:

                st.metric(
                    "Prediction confidence",
                    f"{confidence:.1%}",
                )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        if result.get(
            "class_probabilities"
        ):

            st.markdown(
                "#### Prediction overview"
            )

            probability_cols = st.columns(
                len(
                    result[
                        "class_probabilities"
                    ]
                )
            )

            for index, (
                class_name,
                probability,
            ) in enumerate(
                result[
                    "class_probabilities"
                ].items()
            ):

                with probability_cols[index]:

                    st.metric(
                        class_name,
                        f"{probability:.1%}",
                    )

        # ----------------------------------------------------
        # GEMINI
        # ----------------------------------------------------

        explanation = explain_prediction(
            result,
            text_context,
            gemini_model,
        )

        render_gemini_response(
            explanation,
            result["prediction"],
            gemini_model,
        )

    except Exception as exc:

        st.error(
            f"Prediction error: {type(exc).__name__}: {exc}"
        )


st.divider()
