# Healthcare Disease Prediction — Duplicate & Irrelevant Column Analysis

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">](https://colab.research.google.com/github/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/notebooks/Week_1/02_Duplicates_Irrelevant_Columns.ipynb)

## Notebook

[02_Duplicates_Irrelevant_Columns.ipynb](../../notebooks/Week_1/02_Duplicates_Irrelevant_Columns.ipynb)

## Dataset

[healthcare_dataset.csv](../../dataset/healthcare_dataset.csv)

---

# Healthcare Disease Prediction — Duplicate & Irrelevant Column Analysis

## Objective

The objective of this notebook is to identify duplicate records and evaluate whether any columns may be irrelevant or unsuitable for disease prediction.

This notebook focuses on analysis and documentation. Final column removal and data cleaning will be performed in the appropriate cleaning/preprocessing stage.

---

```python
# Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully.")
```

---

## 1. Load the Dataset

The dataset is loaded directly from the Team-6 GitHub repository.

---

```python
# GitHub raw dataset URL

url = "https://raw.githubusercontent.com/vyasanbmathew2008/Team-6/main/healthcare-disease-prediction/dataset/healthcare_dataset.csv"

# Load dataset
df = pd.read_csv(url)

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
```

---

```python
# Display the first 5 records

df.head()
```

---

## 2. Check Duplicate Records

Duplicate records can cause problems during model training because the same observation may appear more than once.

We first check for exact duplicate rows across all columns.

---

```python
# Count exact duplicate rows

duplicate_count = df.duplicated().sum()

print(f"Number of exact duplicate rows: {duplicate_count}")
```

---

```python
# Calculate duplicate percentage

duplicate_percentage = (duplicate_count / len(df)) * 100

print(f"Percentage of duplicate rows: {duplicate_percentage:.2f}%")
```

---

## 3. Inspect Duplicate Records

If duplicate records are present, we display all copies of those records for further inspection.

---

```python
# Display duplicate records

if duplicate_count > 0:
    duplicate_rows = df[df.duplicated(keep=False)]
    display(duplicate_rows)
else:
    print("No exact duplicate records were found.")
```

---

## 4. Check Repeated Patient Names

Repeated names do not automatically mean duplicate records. A patient may have multiple admissions.

Therefore, repeated names are inspected only for analysis and are not automatically removed.

---

```python
# Check repeated patient names

if "Name" in df.columns:
    repeated_names = df["Name"].value_counts()
    repeated_names = repeated_names[repeated_names > 1]

    print(f"Number of names appearing more than once: {len(repeated_names)}")
    display(repeated_names.head(20).to_frame("Record Count"))
else:
    print("Name column not found.")
```

---

## 5. Identify Potentially Irrelevant Columns

A feature should not be removed simply because it has many unique values.

Each column should be evaluated based on:

- Its relevance to disease prediction
- Whether it represents an identifier
- Whether it contains useful predictive information
- Whether it creates unnecessary complexity
- Whether it could introduce data leakage

---

```python
# Display columns, data types, and number of unique values

column_analysis = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Unique Values": [df[column].nunique(dropna=True) for column in df.columns],
    "Unique Percentage": [
        (df[column].nunique(dropna=True) / len(df)) * 100
        for column in df.columns
    ]
})

column_analysis
```

---

## 6. Initial Feature Relevance Review

| Column | Initial Assessment | Reason |
|---|---|---|
| Name | Potentially irrelevant | Identifies a person and is high-cardinality; unlikely to provide generalizable disease information. |
| Age | Relevant | Demographic information that may be predictive of disease. |
| Gender | Potentially relevant | Demographic feature that may contain predictive information. |
| Blood Type | Potentially relevant | Can be investigated as a possible predictor. |
| Medical Condition | Target | The disease/condition to be predicted. |
| Date of Admission | Potentially relevant | Can be transformed into useful time-based features if justified. |
| Doctor | Potentially irrelevant/high-cardinality | May identify individual doctors rather than patient characteristics. |
| Hospital | Potentially irrelevant/high-cardinality | May capture institution-specific patterns rather than generalizable patient information. |
| Insurance Provider | Potentially relevant | Can be investigated as a categorical feature. |
| Billing Amount | Potentially relevant | Numerical feature that may contain useful information, subject to domain/leakage considerations. |
| Room Number | Potentially irrelevant | Usually represents a location/administrative identifier rather than patient health information. |
| Admission Type | Potentially relevant | May contain useful information about the nature of admission. |
| Discharge Date | Potentially relevant | Can potentially be transformed into length of stay, but may cause leakage depending on prediction timing. |
| Medication | Potentially relevant | May contain information associated with the condition, depending on when prediction is made. |
| Test Results | Potentially relevant | May contain useful predictive information, depending on when results are available. |

> These are initial assessments, not final removal decisions. The team should confirm the prediction scenario before removing features.

---

## 7. Check the Target Variable

`Medical Condition` is the target variable for this project.

We inspect its categories and distribution to confirm the disease classes present in the dataset.

---

```python
# Check target variable

target_column = "Medical Condition"

if target_column in df.columns:
    print("Target variable:", target_column)
    print("\nTarget categories:")
    print(df[target_column].unique())
else:
    print(f"Target column '{target_column}' was not found.")
```

---

```python
# Target class distribution

if target_column in df.columns:
    target_counts = df[target_column].value_counts()
    target_percentages = df[target_column].value_counts(normalize=True) * 100

    target_summary = pd.DataFrame({
        "Count": target_counts,
        "Percentage": target_percentages.round(2)
    })

    display(target_summary)
```

---

## 8. Visualize Target Distribution

A bar chart is used to understand the distribution of the disease classes.

---

```python
# Plot target distribution

if target_column in df.columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=target_column)
    plt.title("Distribution of Medical Conditions")
    plt.xlabel("Medical Condition")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

---

## 9. Duplicate and Irrelevant Column Summary

---

```python
# Generate analysis summary

print("DUPLICATE & IRRELEVANT COLUMN ANALYSIS")
print("=" * 55)
print(f"Total records            : {len(df)}")
print(f"Total columns            : {len(df.columns)}")
print(f"Exact duplicate records  : {duplicate_count}")
print(f"Duplicate percentage     : {duplicate_percentage:.2f}%")
print(f"Target variable          : {target_column}")

if target_column in df.columns:
    print(f"Number of target classes : {df[target_column].nunique()}")
```
