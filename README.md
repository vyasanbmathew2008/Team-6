# Healthcare Disease Prediction

### Machine Learning & Generative AI Project

A team-based healthcare machine-learning project focused on predicting a patient's **Medical Condition** from structured healthcare data, evaluating multiple ML models, deploying the best-performing solution, and enhancing the application with Generative AI.

> **Disclaimer:** This project is intended for educational and research purposes only. Predictions and AI-generated explanations must not be treated as a medical diagnosis or a substitute for advice from a qualified healthcare professional.

---

## Project Overview

The project uses a healthcare dataset containing patient demographics, admission information, medical history, billing information, medications, and test results.

### Prediction Task

**Input:** Relevant patient and healthcare features

**Output:** `Medical Condition`

The target variable represents disease/condition classes such as **Cancer, Diabetes, Obesity, Asthma**, and other conditions present in the dataset.

The project follows a complete machine-learning workflow:

**Data Collection → Data Understanding → Data Cleaning → EDA → Preprocessing → Model Training → Evaluation → Best Model Selection → Deployment → GenAI Enhancement**

---

## Dataset

### Healthcare Dataset Features

The original dataset contains fields including:

- `Name`
- `Age`
- `Gender`
- `Blood Type`
- `Medical Condition` — **Target Variable**
- `Date of Admission`
- `Doctor`
- `Hospital`
- `Insurance Provider`
- `Billing Amount`
- `Room Number`
- `Admission Type`
- `Discharge Date`
- `Medication`
- `Test Results`

### Data Sources

- **Original Dataset:** [Hugging Face Healthcare Dataset](https://huggingface.co/datasets/11andrea2233/healthcare_dataset)
- **Project Dataset:** `healthcare-disease-prediction/dataset/healthcare_dataset.csv`
- **Processed Dataset:** Will be added after completion of the preprocessing pipeline.

---

## Project Objectives

### 1. Collect & Explore

- Understand the dataset structure and features.
- Check data types.
- Identify missing values.
- Identify duplicate records.
- Identify irrelevant columns.
- Analyze categorical and numerical features.
- Define `Medical Condition` as the prediction target.

### 2. Systematic Preprocessing

Build a reusable and reliable preprocessing pipeline that includes:

- Handling missing values.
- Removing duplicate records.
- Correcting inconsistent text formatting.
- Converting and extracting useful information from date fields.
- Encoding categorical variables.
- Processing numerical variables.
- Detecting and handling outliers where appropriate.
- Feature selection and feature engineering.
- Splitting data into training and testing sets.
- Preparing the final ML-ready dataset.

### 3. Model Development

Train and compare multiple machine-learning algorithms for `Medical Condition` prediction, including suitable classification models such as:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes
- Gradient Boosting

### 4. Evaluate & Compare

Evaluate the trained models using standardized metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC, where applicable

The strongest model will be selected based on the evaluation results and project requirements.

### 5. Deployment

Deploy the selected model through a user-friendly application. The planned interface is **Streamlit**, with an API/backend approach such as FastAPI or Flask considered where appropriate.

### 6. GenAI Enhancement

Integrate the **Gemini API** to provide:

- Simple explanations of model predictions.
- General information about the predicted medical condition.
- Contextual health-related information.
- User-friendly explanations of prediction results.
- Guidance on when a user should consult a qualified healthcare professional.

GenAI output will be presented as supportive educational information and **not as medical diagnosis or treatment advice**.

---

## Team Work Distribution — Week 1

The initial dataset analysis is divided into **8 focused topics** among the five team members.

| Topic | Responsibility | Status |
|---|---|---|
| 1. Data Types | Inspect and document column data types and identify type-related issues. | ⬜ |
| 2. Missing Values | Identify, summarize, and document missing-value patterns. | ⬜ |
| 3. Duplicate Records | Detect and analyze duplicate rows. | ⬜ |
| 4. Irrelevant Columns | Identify columns that are unnecessary or unsuitable for prediction. | ⬜ |
| 5. Categorical & Numerical Feature Analysis | Analyze distributions, unique values, and statistical characteristics of feature types. | ⬜ |
| 6. Exploratory Data Analysis (EDA) | Explore relationships, distributions, trends, and important patterns using visualizations. | ⬜ |
| 7. Initial Data Cleaning | Apply initial cleaning such as text standardization, formatting fixes, and basic data-quality corrections. | ⬜ |
| 8. Document Dataset Characteristics | Consolidate dataset findings, decisions, assumptions, and observations. | ⬜ |

### Suggested Team Allocation

- **Person 1:** Data Types + Missing Values
- **Person 2:** Duplicate Records + Irrelevant Columns
- **Person 3:** Categorical & Numerical Feature Analysis
- **Person 4:** Exploratory Data Analysis (EDA)
- **Person 5:** Initial Data Cleaning + Documentation

The team will merge these individual contributions into the overall preprocessing workflow.

---

## Week 1 — Dataset Discovery & Initial Analysis

**Goal:** Understand the dataset, identify data-quality issues, perform EDA, and prepare the foundation for preprocessing.

### Tasks

- [x] Select and download the healthcare dataset
- [x] Understand dataset structure and features
- [ ] Check data types
- [ ] Identify missing values
- [ ] Identify duplicate records
- [ ] Identify irrelevant columns
- [ ] Analyze categorical and numerical features
- [ ] Perform EDA
- [ ] Perform initial data cleaning
- [ ] Document dataset characteristics

### Week 1 Deliverables

- [x] Original healthcare dataset
- [ ] Dataset description
- [ ] Data types and missing-value analysis
- [ ] Duplicate and irrelevant-column analysis
- [ ] Categorical and numerical feature analysis
- [ ] EDA report/visualizations
- [ ] Initial data-cleaning notebook/script
- [ ] Cleaned dataset
- [ ] Initial preprocessing documentation

### Notebook Organization

Recommended notebook for the first assignment:

`01_Data_Types_Missing_Values.ipynb`

**Notebook title:** `Healthcare Disease Prediction — Data Types & Missing Value Analysis`

Additional notebooks can follow the same numbered naming convention as the project grows.

---

## Week 2 — Complete Preprocessing & ML Modelling

**Goal:** Build the final preprocessing pipeline and train multiple classification models.

### Preprocessing

- [ ] Handle missing values
- [ ] Remove duplicate records
- [ ] Standardize inconsistent data
- [ ] Encode categorical variables
- [ ] Scale numerical features where required
- [ ] Detect and handle outliers
- [ ] Perform feature selection
- [ ] Perform feature engineering
- [ ] Split data into training and testing sets
- [ ] Prepare final ML-ready data
- [ ] Build a reusable preprocessing pipeline

### Machine Learning

- [ ] Train Logistic Regression
- [ ] Train Decision Tree
- [ ] Train Random Forest
- [ ] Train KNN
- [ ] Train SVM
- [ ] Train Naive Bayes
- [ ] Train Gradient Boosting
- [ ] Compare model performance
- [ ] Select the best-performing model

### Week 2 Deliverables

- [ ] Fully preprocessed dataset
- [ ] Reusable preprocessing pipeline
- [ ] Trained classification models
- [ ] Model comparison results
- [ ] Selected best model

---

## Week 3 — Evaluation, Deployment & GenAI

**Goal:** Evaluate the final models, deploy the prediction system, and integrate Generative AI.

### Model Evaluation

- [ ] Calculate accuracy
- [ ] Calculate precision
- [ ] Calculate recall
- [ ] Calculate F1-score
- [ ] Generate confusion matrix
- [ ] Evaluate ROC-AUC where applicable
- [ ] Analyze model strengths and weaknesses
- [ ] Finalize the best model

### Deployment

- [ ] Serialize the selected model
- [ ] Build the prediction interface
- [ ] Integrate the trained model into the application
- [ ] Implement user input validation
- [ ] Test the end-to-end prediction workflow
- [ ] Deploy the application/API to a suitable platform

### GenAI

- [ ] Integrate Gemini API
- [ ] Generate explanations for model predictions
- [ ] Provide general information about predicted conditions
- [ ] Add contextual healthcare information
- [ ] Build an interactive AI assistant where appropriate
- [ ] Include clear medical-safety disclaimers

### Week 3 Deliverables

- [ ] Evaluated final model
- [ ] Working prediction system
- [ ] Deployed application/API
- [ ] Gemini/GenAI integration
- [ ] Final project documentation

---

## Project Structure

```text
Team-6/
│
├── healthcare-disease-prediction/
│   │
│   ├── dataset/
│   │   ├── healthcare_dataset.csv
│   │   └── preprocessed_data.csv
│   │
│   ├── notebooks/
│   │   ├── 01_Data_Types_Missing_Values.ipynb
│   │   ├── 02_Duplicate_Irrelevant_Columns.ipynb
│   │   ├── 03_Categorical_Numerical_Analysis.ipynb
│   │   ├── 04_EDA.ipynb
│   │   ├── 05_Initial_Data_Cleaning.ipynb
│   │   └── ...
│   │
│   ├── models/
│   │   └── best_model.pkl
│   │
│   ├── src/
│   │   ├── preprocessing.py
│   │   ├── training.py
│   │   └── prediction.py
│   │
│   ├── app/
│   │   └── app.py
│   │
│   ├── requirements.txt
│   └── README.md
│
└── README.md
```

> The structure above is the planned project structure. Files and folders may be added or renamed as implementation progresses.

---

## Technology Stack

| Category | Technologies |
|---|---|
| **Programming Language** | Python |
| **Development Environment** | Google Colab, VS Code (where applicable) |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Deployment** | Streamlit |
| **Backend/API** | FastAPI or Flask, where required |
| **Generative AI** | Gemini API |
| **Version Control** | Git, GitHub |
|
---

## Project Workflow

```text
Healthcare Dataset
       ↓
Data Understanding
       ↓
Data Quality Analysis
       ↓
Initial Data Cleaning
       ↓
Exploratory Data Analysis (EDA)
       ↓
Complete Preprocessing
       ↓
Feature Engineering & Selection
       ↓
Train / Test Split
       ↓
Multiple ML Models
       ↓
Model Evaluation & Comparison
       ↓
Best Model Selection
       ↓
Deployment
       ↓
Gemini GenAI Enhancement
       ↓
Final Healthcare Disease Prediction System
```

---

## Current Project Status

### Overall Progress

| Phase | Status |
|---|---|
| Dataset Collection | ✅ Completed |
| Dataset Exploration | 🔄 In Progress |
| Initial Data Cleaning | ⬜ Pending |
| Complete Preprocessing | ⬜ Pending |
| ML Model Development | ⬜ Pending |
| Model Evaluation | ⬜ Pending |
| Deployment | ⬜ Pending |
| GenAI Integration | ⬜ Pending |
| Final Documentation | ⬜ Pending |

---

## Important Data-Quality Considerations

During initial analysis and cleaning, the team will specifically check for:

- Missing or null values.
- Duplicate patient records.
- Inconsistent capitalization and text formatting.
- Invalid or inconsistent dates.
- Incorrect numerical data types.
- Unusual or potentially invalid numerical values.
- High-cardinality categorical columns.
- Columns that may cause data leakage or provide little predictive value.
- Class distribution of the `Medical Condition` target.

Text normalization should standardize formatting without changing the underlying meaning of the data. For example, inconsistent capitalization in names can be corrected during initial cleaning, while true spelling errors should not be assumed or silently changed.

---

## Reproducibility

To reproduce the project locally:

1. Clone the repository.
2. Install the required Python dependencies from `requirements.txt`.
3. Run the notebooks in order.
4. Generate the processed dataset.
5. Train and evaluate the models.
6. Save the selected model.
7. Run the Streamlit application.

Environment-specific setup instructions will be added as the project implementation is finalized.

---

## Disclaimer

This project is an academic/educational machine-learning implementation. It does not provide medical diagnosis, treatment, or professional healthcare advice. Any prediction generated by the model or explanation generated by the GenAI component should be interpreted cautiously and discussed with a qualified healthcare professional when relevant.

---

## License

See the repository license file for applicable licensing information.
