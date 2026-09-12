# Healthcare Disease Prediction

### Machine Learning & Generative AI Internship Project

> A three-week implementation roadmap for building, evaluating, deploying, and enhancing a healthcare disease-prediction system.

---

## Project Snapshot

| Field | Details |
| --- | --- |
| **Duration** | 3 weeks |
| **Domain** | Healthcare / Medicine |
| **Focus** | Machine Learning & Generative AI |
| **Primary Outcome** | A deployed healthcare disease-prediction system with GenAI-assisted explanations |

> **Important:** This system is intended for educational and research purposes only. It must not be used as a substitute for professional medical diagnosis or treatment.

---

## Dataset & Data Files

- **Original Dataset:** [https://huggingface.co/datasets/11andrea2233/healthcare_dataset]
- **GitHub Dataset (CSV):** [https://github.com/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/dataset/healthcare_dataset.csv]
- **Cleaned Dataset:** [[Cleaned Dataset URL](https://raw.githubusercontent.com/vyasanbmathew2008/Team-6/refs/heads/main/healthcare-disease-prediction/dataset/cleaned_data.csv)]
- **Preprocessed Dataset:** [Link]


---

Project Objectives

- [x] Collect & Explore: Gather and understand the structure of the healthcare dataset, identify data types, missing values, duplicates, categorical and numerical features, and define "Medical Condition" as the target variable for disease prediction.

- [x] Systematic Preprocessing: Build a robust data-cleaning and preprocessing pipeline, including handling missing values, duplicates, inconsistent text formatting, date conversion, categorical encoding, numerical feature processing, and appropriate feature selection.

- [ ] Model Development: Train and compare multiple machine-learning algorithms to predict the patient's "Medical Condition" from relevant healthcare and admission features.

- [ ] Evaluate & Compare: Benchmark the trained models using standardized evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix, then select the strongest model.

- [ ] Deployment: Deploy the selected disease-prediction model through a user-friendly Streamlit web application with an appropriate prediction interface.

- [ ] GenAI Enhancement: Integrate the Gemini API to provide user-friendly explanations of the predicted disease, relevant contextual information, and appropriate healthcare recommendations while clearly presenting the output as supportive information rather than a medical diagnosis.

---

## Sample Project Structure

```text
healthcare-disease-prediction/
│
├── dataset/
│   ├── healthcare_dataset.csv
│   └── preprocessed_data.csv
│
├── notebooks/
│   │
│   ├── Week_1/
│   │   ├── 01_Data_Types_Missing_Values.ipynb
│   │   ├── 02_Duplicates_Irrelevant_Columns.ipynb
│   │   ├── 03_Categorical_Numerical_Analysis.ipynb
│   │   ├── 04_Exploratory_Data_Analysis.ipynb
│   │   └── 05_Initial_Data_Preprocessing_Dataset_Documentation.ipynb
│   │
│   ├── Week_2/
│   │   └── week2_modelling.ipynb
│   │
│   └── Week_3/
│       └── week3_evaluation.ipynb
│
├── Description/
│   │
│   ├── Week_1/
│   │   ├── 01_Data_Types_Missing_Values.md
│   │   ├── 02_Duplicates_Irrelevant_Columns.md
│   │   ├── 03_Categorical_Numerical_Analysis.md
│   │   ├── 04_Exploratory_Data_Analysis.md
│   │   └── 05_Initial_Data_Preprocessing_Dataset_Documentation.md
│   │
│   ├── Week_2/
│   │   └── week2_modelling.md
│   │
│   └── Week_3/
│       └── week3_evaluation.md
│
├── models/
│   └── best_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   └── prediction.py
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Three-Week Execution Plan

### Week 1 — Dataset Discovery & Initial Preprocessing

**Objective:** Understand the healthcare dataset, identify data-quality issues, and produce an initial cleaned version.

---

## 📚 Week 1 Notebooks

| No.    | Notebook                                                                                                                                                                                                                                                                         | Documentation                                                                                                                                                                                                                                                                                |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **01** | [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">](https://colab.research.google.com/github/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/notebooks/Week_1/01_Data_Types_Missing_Values.ipynb)                        | [<img src="https://img.shields.io/badge/GitHub-Documentation-181717?logo=github&logoColor=white" alt="GitHub Documentation">](https://github.com/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/Description/Week_1/01_Data_Types_Missing_Values.md)                        |
| **02** | [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">](https://colab.research.google.com/github/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/notebooks/Week_1/02_Duplicates_Irrelevant_Columns.ipynb)                    | [<img src="https://img.shields.io/badge/GitHub-Documentation-181717?logo=github&logoColor=white" alt="GitHub Documentation">](https://github.com/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/Description/Week_1/02_Duplicates_Irrelevant_Columns.md)                    |
| **03** | [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">](https://colab.research.google.com/github/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/notebooks/Week_1/03_Categorical_Numerical_Analysis.ipynb)                   | [<img src="https://img.shields.io/badge/GitHub-Documentation-181717?logo=github&logoColor=white" alt="GitHub Documentation">](https://github.com/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/Description/Week_1/03_Categorical_Numerical_Analysis.md)                   |
| **04** | [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">](https://colab.research.google.com/github/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/notebooks/Week_1/04_Exploratory_Data_Analysis.ipynb)                        | [<img src="https://img.shields.io/badge/GitHub-Documentation-181717?logo=github&logoColor=white" alt="GitHub Documentation">](https://github.com/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/Description/Week_1/04_Exploratory_Data_Analysis.md)                        |
| **05** | [<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">](https://colab.research.google.com/github/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/notebooks/Week_1/05_Initial_Data_Preprocessing_Dataset_Documentation.ipynb) | [<img src="https://img.shields.io/badge/GitHub-Documentation-181717?logo=github&logoColor=white" alt="GitHub Documentation">](https://github.com/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/Description/Week_1/05_Initial_Data_Preprocessing_Dataset_Documentation.md) |



#### Tasks

- [x] Select and download the healthcare dataset
- [x] Understand the dataset structure and features
- [x] Check data types & Identify missing values
- [x] Identify duplicate records & irrelevant columns
- [x] Analyze categorical and numerical features
- [x] Perform exploratory data analysis (EDA)
- [x] Perform initial data preprocessing & Document dataset characteristics

#### Week 1 Deliverables

- [x] Original dataset
- [x] Dataset description
- [x] Initial EDA report
- [x] Initial preprocessing notebook or script
- [x] Initial preprocessing documentation

---

### Week 2 — Complete Preprocessing & ML Modelling

**Objective:** Build the final preprocessing pipeline and train multiple machine-learning models for comparison.

#### Data Preprocessing Tasks

- [ ] Handle missing values
- [ ] Remove duplicate records
- [ ] Handle inconsistent data
- [ ] Encode categorical variables
- [ ] Scale numerical features where required
- [ ] Detect and handle outliers
- [ ] Perform feature selection
- [ ] Perform feature engineering
- [ ] Split the data into training and testing sets
- [ ] Prepare the final ML-ready dataset

#### Machine Learning Modelling Tasks

Train and compare suitable algorithms, including:

- [ ] Logistic Regression
- [ ] Decision Tree
- [ ] Random Forest
- [ ] K-Nearest Neighbors (KNN)
- [ ] Support Vector Machine (SVM)
- [ ] Naive Bayes
- [ ] Gradient Boosting

#### Week 2 Deliverables

- [ ] Fully preprocessed dataset
- [ ] Reusable preprocessing pipeline
- [ ] Trained machine-learning models
- [ ] Model comparison report
- [ ] Selected best-performing model

---

### Week 3 — Evaluation, Deployment & GenAI

**Objective:** Evaluate the models, deploy the selected solution, and add useful GenAI capabilities.

#### Model Evaluation Tasks

Evaluate the trained models using appropriate metrics:

- [ ] Accuracy
- [ ] Precision
- [ ] Recall
- [ ] F1-score
- [ ] Confusion matrix
- [ ] ROC-AUC, where applicable
- [ ] Select the best-performing model based on evaluation results

#### Deployment Tasks

- [ ] Serialize the selected model
- [ ] Build a backend API using FastAPI or Flask
- [ ] Create a user-input interface
- [ ] Implement the prediction system
- [ ] Test the API and prediction workflow
- [ ] Deploy to a suitable cloud platform

#### GenAI Implementation Tasks

- [ ] Explain prediction results in simple language
- [ ] Provide general information about predicted diseases
- [ ] Generate health-related explanations based on model output
- [ ] Create an interactive AI assistant
- [ ] Provide guidance on when to consult an appropriate healthcare professional

#### Week 3 Deliverables

- [ ] Evaluated ML model
- [ ] Final prediction system
- [ ] Deployed application or API
- [ ] GenAI integration
- [ ] Final project documentation

---

## Project Workflow

- [ ] Original dataset
- [ ] Initial data preprocessing
- [ ] Complete data preprocessing
- [ ] Feature engineering
- [ ] ML model training
- [ ] Model evaluation
- [ ] Best model selection
- [ ] Model deployment
- [ ] GenAI integration
- [ ] Final healthcare prediction system

---

## Technology Stack

| Category | Technologies |
| --- | --- |
| **Languages & Core** | Python, Google Colab Notebook |
| **Data & Analytics** | Pandas, NumPy, Matplotlib, Seaborn |
| **Machine Learning** | N/A |
| **Backend & Integration** | Generative AI API |
| **Version Control** | Git and GitHub |
