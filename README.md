# Team 6 - Healthcare AI Project

A student machine-learning project for predicting heart disease, diabetes and infectious diseases.

## Datasets

- Heart disease
- Diabetes
- Infectious disease / symptoms

Lung disease and lung cancer are not included in this version.

## Files

- `train.py` - trains the models and saves them in `models/`
- `app.py` - Streamlit interface
- `data/raw/` - project datasets

## Run

Install the packages:

    pip install -r requirements.txt

Train the models:

    python train.py

Start the app:

    streamlit run app.py

For the optional Gemini explanation, create a local `.env` file and add your API key. Do not commit the key.

This is an educational project and its predictions are not medical diagnoses.
