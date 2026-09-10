# Healthcare Disease Prediction — Data Types & Missing Value Analysis

## Notebook

This document provides the documentation for the **Data Types & Missing Value Analysis** performed in Week 1.

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">](https://colab.research.google.com/github/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/notebooks/Week_1/01_Data_Types_Missing_Values.ipynb)

---

## 1. Objective

The objective of this analysis is to understand the structure and quality of the healthcare dataset before applying machine learning preprocessing techniques.

The analysis focuses on:

- Inspecting the dataset structure
- Checking the number of rows and columns
- Identifying column names
- Identifying data types
- Identifying numerical and categorical features
- Identifying date-related columns
- Checking unique values
- Checking missing values
- Calculating missing-value percentages
- Visualizing missing values
- Documenting potential data-quality issues

No final data-cleaning operations are performed in this analysis.

---

## 2. Dataset

The dataset used in this analysis is the healthcare dataset stored in the project repository.

**Dataset:** [`healthcare_dataset.csv`](../../dataset/healthcare_dataset.csv)

The dataset contains patient/admission-related information such as:

- Name
- Age
- Gender
- Blood Type
- Medical Condition
- Date of Admission
- Doctor
- Hospital
- Insurance Provider
- Billing Amount
- Room Number
- Admission Type
- Discharge Date
- Medication
- Test Results

The target variable for the project is:

**Medical Condition**

---

## 3. Dataset Structure

The first stage of the analysis checks the overall dimensions and structure of the dataset.

The following information is examined:

- Number of records
- Number of columns
- Column names
- Non-null values
- Data types
- Memory usage

The `head()` and `tail()` functions are also used to inspect sample records.

---

## 4. Data Types

The data type of every column is examined to determine whether the column contains numerical, categorical/text, or date-related information.

### Numerical Features

Numerical features contain values that can be represented as numbers.

Examples include:

- Age
- Billing Amount
- Room Number

### Categorical/Text Features

Categorical or text features contain labels or textual information.

Examples include:

- Gender
- Blood Type
- Medical Condition
- Admission Type
- Medication
- Test Results
- Insurance Provider

Other text columns, such as Name, Doctor, and Hospital, contain many unique values and require additional consideration during feature selection.

### Date Features

The dataset contains:

- Date of Admission
- Discharge Date

These columns may initially be stored as `object` when loaded from CSV. They should be converted to an appropriate datetime format during the cleaning/preprocessing stage.

---

## 5. Unique Value Analysis

The number of unique values in every column is examined.

This helps identify:

- Categorical variables
- Low-cardinality features
- High-cardinality features
- Potential identifier-like columns

A high number of unique values does not automatically mean that a column should be removed. Feature relevance must be considered before making that decision.

---

## 6. Missing Value Analysis

Missing values are checked for every column using:

```python
df.isnull().sum()
