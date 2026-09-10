# 4) Exploratory Data Analysis (EDA)

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">](https://colab.research.google.com/github/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/notebooks/Week_1/04_Exploratory_Data_Analysis.ipynb)

## Notebook

[04_Exploratory_Data_Analysis.ipynb](../../notebooks/Week_1/04_Exploratory_Data_Analysis.ipynb)

## Dataset

[healthcare_dataset.csv](../../dataset/healthcare_dataset.csv)

---

## Objective

The objective of this notebook is to perform **Exploratory Data Analysis (EDA)** on the healthcare disease prediction dataset.

The analysis focuses on:

* Dataset structure
* Missing values
* Target variable distribution
* Numerical feature distributions
* Categorical feature distributions
* Relationships between features and medical conditions
* Correlations between numerical variables
* Potential outliers
* Important data-quality considerations

No permanent data cleaning or feature transformation is performed in this notebook.

---

## 1. Import Required Libraries

The following Python libraries are used for data analysis and visualization.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 2. Load Dataset

The healthcare dataset is loaded directly from the GitHub repository.

```python
url = "https://raw.githubusercontent.com/vyasanbmathew2008/Team-6/main/healthcare-disease-prediction/dataset/healthcare_dataset.csv"

df = pd.read_csv(url)

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")
```

---

## 3. Dataset Overview

The first five records are displayed to understand the structure of the dataset.

```python
df.head()
```

The dataset information is then examined.

```python
df.info()
```

Descriptive statistics are generated for the dataset.

```python
df.describe(include="all").T
```

This provides an initial understanding of the dataset's columns, data types, ranges, and distributions.

---

## 4. Missing Value Analysis

Missing values are examined using counts and percentages.

```python
missing_values = df.isnull().sum()

missing_percentage = (
    df.isnull().mean() * 100
)

missing_summary = pd.DataFrame({
    "Missing Values": missing_values,
    "Missing Percentage": missing_percentage
})

missing_summary
```

A heatmap is used to visually inspect missing values.

```python
plt.figure(figsize=(12, 5))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False
)

plt.title("Missing Values Heatmap")
plt.xlabel("Columns")
plt.ylabel("Records")
plt.tight_layout()
plt.show()
```

No missing values are permanently filled or removed in this notebook.

---

## 5. Target Variable Analysis

The target variable for the project is:

**Medical Condition**

The target classes are examined using frequency counts.

```python
target_column = "Medical Condition"

target_counts = df[target_column].value_counts()

target_counts
```

The percentage distribution of each target class is calculated.

```python
target_percentages = (
    df[target_column]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

target_distribution = pd.DataFrame({
    "Count": target_counts,
    "Percentage": target_percentages
})

target_distribution
```

The target distribution is visualized using a count plot.

```python
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x=target_column,
    order=df[target_column].value_counts().index
)

plt.title("Distribution of Medical Conditions")
plt.xlabel("Medical Condition")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

This analysis helps determine whether the disease classes are approximately balanced.

---

## 6. Numerical Feature Analysis

Numerical features are identified automatically.

```python
numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print("Numerical Columns:")

for column in numerical_columns:
    print("-", column)
```

Descriptive statistics are generated for numerical features.

```python
df[numerical_columns].describe().T
```

Histograms are created to examine numerical feature distributions.

```python
for column in numerical_columns:

    plt.figure(figsize=(8, 4))

    sns.histplot(
        data=df,
        x=column,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
```

Box plots are used to identify potential outliers.

```python
for column in numerical_columns:

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        data=df,
        x=column
    )

    plt.title(f"Box Plot of {column}")
    plt.xlabel(column)

    plt.tight_layout()
    plt.show()
```

---

## 7. Age Analysis

Age is analyzed because it may have a relationship with different medical conditions.

The overall age distribution is visualized below.

```python
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Age",
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
```

Age is then compared across medical conditions.

```python
plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x="Medical Condition",
    y="Age"
)

plt.title("Age by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Age")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

This helps identify whether different disease classes have noticeably different age distributions.

---

## 8. Billing Amount Analysis

Billing Amount is analyzed as a numerical feature.

```python
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Billing Amount",
    kde=True
)

plt.title("Billing Amount Distribution")
plt.xlabel("Billing Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
```

Billing Amount is also compared across medical conditions.

```python
plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x="Medical Condition",
    y="Billing Amount"
)

plt.title("Billing Amount by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Billing Amount")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

Billing Amount should be reviewed later for possible data leakage depending on the intended prediction stage.

---

## 9. Categorical Feature Analysis

Categorical/text columns are identified automatically.

```python
categorical_columns = df.select_dtypes(
    include="object"
).columns.tolist()

print("Categorical/Text Columns:")

for column in categorical_columns:
    print("-", column)
```

Frequency counts are displayed for each categorical feature.

```python
for column in categorical_columns:

    print("\n" + "=" * 60)
    print(f"{column}")
    print("=" * 60)

    print(df[column].value_counts(dropna=False))
```

This helps identify the distribution and cardinality of categorical variables.

---

## 10. Gender Analysis

Gender distribution is visualized using a count plot.

```python
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Gender",
    order=df["Gender"].value_counts().index
)

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
```

Gender is compared with Medical Condition.

```python
gender_condition = pd.crosstab(
    df["Gender"],
    df["Medical Condition"],
    normalize="index"
) * 100

gender_condition.round(2)
```

---

## 11. Admission Type Analysis

Admission Type is examined to understand the distribution of admission categories.

```python
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Admission Type",
    order=df["Admission Type"].value_counts().index
)

plt.title("Admission Type Distribution")
plt.xlabel("Admission Type")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

Admission Type is compared with Medical Condition.

```python
admission_condition = pd.crosstab(
    df["Admission Type"],
    df["Medical Condition"],
    normalize="index"
) * 100

admission_condition.round(2)
```

---

## 12. Blood Type Analysis

Blood Type distribution is visualized.

```python
plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Blood Type",
    order=df["Blood Type"].value_counts().index
)

plt.title("Blood Type Distribution")
plt.xlabel("Blood Type")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
```

Blood Type is compared with Medical Condition.

```python
blood_condition = pd.crosstab(
    df["Blood Type"],
    df["Medical Condition"],
    normalize="index"
) * 100

blood_condition.round(2)
```

---

## 13. Insurance Provider Analysis

Insurance Provider distribution is visualized.

```python
plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="Insurance Provider",
    order=df["Insurance Provider"].value_counts().index
)

plt.title("Insurance Provider Distribution")
plt.xlabel("Insurance Provider")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

Insurance Provider is compared with Medical Condition.

```python
insurance_condition = pd.crosstab(
    df["Insurance Provider"],
    df["Medical Condition"],
    normalize="index"
) * 100

insurance_condition.round(2)
```

---

## 14. Correlation Analysis

Correlation analysis is performed on numerical variables.

```python
correlation_matrix = df[numerical_columns].corr()

correlation_matrix
```

The correlation matrix is visualized using a heatmap.

```python
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.show()
```

The correlation matrix helps identify potentially related numerical variables.

---

## 15. Numerical Features vs Target

Numerical features are compared across Medical Condition classes.

```python
for column in numerical_columns:

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="Medical Condition",
        y=column
    )

    plt.title(f"{column} by Medical Condition")
    plt.xlabel("Medical Condition")
    plt.ylabel(column)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
```

These plots help determine whether numerical features have visibly different distributions across disease classes.

---

## 16. Medication Analysis

Medication is analyzed because it may contain information about disease patterns.

The distribution of medications is visualized.

```python
plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="Medication",
    order=df["Medication"].value_counts().index
)

plt.title("Medication Distribution")
plt.xlabel("Medication")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

Medication is compared with Medical Condition.

```python
medication_condition = pd.crosstab(
    df["Medication"],
    df["Medical Condition"],
    normalize="index"
) * 100

medication_condition.round(2)
```

Medication should be evaluated according to the intended prediction timing because it may be information obtained after diagnosis or treatment.

---

## 17. Test Results Analysis

Test Results are analyzed using a frequency distribution.

```python
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Test Results",
    order=df["Test Results"].value_counts().index
)

plt.title("Test Results Distribution")
plt.xlabel("Test Results")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

Test Results are compared with Medical Condition.

```python
test_condition = pd.crosstab(
    df["Test Results"],
    df["Medical Condition"],
    normalize="index"
) * 100

test_condition.round(2)
```

Test Results may be useful for prediction, but their use should depend on when the prediction is intended to occur.

---

## 18. Date Feature Analysis

The dataset contains two date-related columns:

* Date of Admission
* Discharge Date

The values are initially inspected without permanently transforming them.

```python
date_columns = [
    "Date of Admission",
    "Discharge Date"
]

for column in date_columns:

    print(f"\n{column}")
    print(df[column].head())
```

During later feature engineering, these columns may be converted to datetime values.

Possible derived features include:

* Admission year
* Admission month
* Admission day
* Length of stay

---

## 19. Data Patterns and Observations

The EDA focuses on identifying:

* Target class distribution
* Age patterns
* Billing amount patterns
* Categorical feature distributions
* Feature-versus-target relationships
* Potential numerical outliers
* Correlations between numerical variables
* High-cardinality categorical variables
* Potential data leakage
* Prediction timing

These observations will guide later data-cleaning and machine-learning decisions.

---

## 20. Dataset EDA Summary

Basic dataset statistics are generated.

```python
print("DATASET EDA SUMMARY")
print("=" * 50)

print(f"Number of records       : {len(df)}")
print(f"Number of columns       : {len(df.columns)}")
print(f"Numerical columns       : {len(numerical_columns)}")
print(f"Categorical columns     : {len(categorical_columns)}")

print(
    f"Target classes          : "
    f"{df[target_column].nunique(dropna=True)}"
)

print(
    f"Duplicate rows          : "
    f"{df.duplicated().sum()}"
)

print(
    f"Total missing values    : "
    f"{df.isnull().sum().sum()}"
)
```

The target distribution is summarized.

```python
print("Medical Condition Distribution")
print("=" * 40)

for condition, count in df[target_column].value_counts().items():

    percentage = (
        count / len(df) * 100
    )

    print(
        f"{condition}: "
        f"{count} records "
        f"({percentage:.2f}%)"
    )
```

---

## 21. Important Data Quality Considerations

The EDA identifies several issues that should be considered during later stages.

### Duplicate Records

Duplicate records should be investigated before model training.

### Missing Values

Missing values should be handled during the data-cleaning and preprocessing stages.

### High-Cardinality Features

Columns such as:

* Doctor
* Hospital

may contain many unique values and may require special encoding or feature-selection decisions.

### Potentially Irrelevant Features

Columns such as:

* Name
* Room Number

may have limited predictive value.

### Date Features

Date of Admission and Discharge Date may require feature extraction.

### Potential Data Leakage

Discharge Date, Medication, Test Results, and Billing Amount should be evaluated according to the intended prediction timing.

For example, if the model is intended to predict a disease at admission, information that becomes available only after admission may introduce data leakage.

---

## 22. EDA Summary

The healthcare dataset was explored through:

* Dataset structure analysis
* Missing value analysis
* Target variable distribution
* Numerical feature distributions
* Categorical feature distributions
* Age analysis
* Billing amount analysis
* Gender analysis
* Admission Type analysis
* Blood Type analysis
* Insurance Provider analysis
* Medication analysis
* Test Results analysis
* Correlation analysis
* Feature-versus-target comparisons
* Date feature inspection
* Data-quality assessment

The analysis provides a foundation for:

* Data cleaning
* Feature selection
* Feature engineering
* Encoding
* Scaling
* Machine learning model development

---

## Conclusion

Exploratory Data Analysis was performed to understand the characteristics, distributions, relationships, and potential issues within the healthcare disease prediction dataset.

The findings from this analysis will guide the subsequent stages of:

* Initial data cleaning
* Feature engineering
* Data preprocessing
* Feature selection
* Model development
* Model evaluation

No permanent changes were made to the original dataset during this notebook.
