
# Healthcare Disease Prediction — Categorical & Numerical Feature Analysis

## Notebook

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">](https://colab.research.google.com/github/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/notebooks/Week_1/03_Categorical_Numerical_Analysis.ipynb)

**Notebook:** [`03_Categorical_Numerical_Analysis.ipynb`](../../notebooks/Week_1/03_Categorical_Numerical_Analysis.ipynb)

**Dataset:** [`healthcare_dataset.csv`](../../dataset/healthcare_dataset.csv)

---

## 1. Objective

The objective of this analysis is to understand the categorical and numerical features of the healthcare dataset before applying machine learning preprocessing techniques.

The analysis focuses on:

- Identifying categorical features
- Identifying numerical features
- Examining unique values
- Analyzing categorical feature distributions
- Calculating numerical descriptive statistics
- Examining numerical feature distributions
- Checking skewness
- Identifying potential outliers
- Analyzing the target variable
- Comparing features with the target variable

No permanent feature removal or transformation is performed in this notebook.

---

## 2. Import Required Libraries

The following libraries are used for data analysis and visualization.

```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
````

---

## 3. Load Dataset

The dataset is loaded directly from the GitHub repository.

```python
# Load dataset directly from GitHub
url = "https://raw.githubusercontent.com/vyasanbmathew2008/Team-6/main/healthcare-disease-prediction/dataset/healthcare_dataset.csv"
df = pd.read_csv(url)

print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")
```

---

## 4. Categorical Feature Analysis

Categorical features contain labels, categories, names, or other textual information.

These features may require encoding before being used by machine learning algorithms.

### Identify Categorical/Text Columns

```python
# Identify categorical/text columns
categorical_columns = df.select_dtypes(include="object").columns.tolist()

print("Categorical/Text Columns:\n")

for column in categorical_columns:
    print("-", column)

print(f"\nTotal categorical/text columns: {len(categorical_columns)}")
```

### Display Unique Values and Frequency Counts

```python
# Display unique values and frequency counts for categorical columns

for column in categorical_columns:
    print(f"\n{column} - Unique Values: {df[column].nunique(dropna=True)}")
    print(df[column].value_counts(dropna=False).head(20))
```

This helps identify the number of categories and the frequency of each category.

### Categorical Feature Cardinality

```python
# Calculate cardinality of categorical features

categorical_summary = pd.DataFrame({
    "Column": categorical_columns,
    "Unique Values": [
        df[column].nunique(dropna=True)
        for column in categorical_columns
    ],
    "Unique Percentage": [
        (df[column].nunique(dropna=True) / len(df)) * 100
        for column in categorical_columns
    ]
})

categorical_summary
```

Cardinality helps identify low-cardinality and high-cardinality categorical variables.

### Low-Cardinality Categorical Features

```python
# Identify low-cardinality categorical columns

low_cardinality_columns = [
    column
    for column in categorical_columns
    if df[column].nunique(dropna=True) <= 10
]

print("Low-Cardinality Categorical Columns:\n")

for column in low_cardinality_columns:
    print("-", column)

print(
    f"\nTotal low-cardinality categorical columns: "
    f"{len(low_cardinality_columns)}"
)
```

### Visualize Categorical Feature Distributions

```python
# Plot distributions of low-cardinality categorical features

for column in low_cardinality_columns:
    plt.figure(figsize=(8, 4))

    sns.countplot(
        data=df,
        x=column,
        order=df[column].value_counts().index
    )

    plt.title(f"Distribution of {column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

These plots help identify the distribution of observations across different categorical groups.

---

## 5. Numerical Feature Analysis

Numerical features contain values represented by numbers.

They can be analyzed using descriptive statistics, distributions, skewness, and potential outlier detection.

### Identify Numerical Columns

```python
# Identify numerical columns

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

print("Numerical Columns:\n")

for column in numerical_columns:
    print("-", column)

print(f"\nTotal numerical columns: {len(numerical_columns)}")
```

### Descriptive Statistics

```python
# Generate descriptive statistics for numerical features

numerical_summary = df[numerical_columns].describe().T

numerical_summary
```

The descriptive statistics include:

* Count
* Mean
* Standard deviation
* Minimum
* 25th percentile
* Median
* 75th percentile
* Maximum

### Check Skewness

```python
# Check skewness of numerical features

skewness = (
    df[numerical_columns]
    .skew(numeric_only=True)
    .sort_values(ascending=False)
)

skewness_df = pd.DataFrame({
    "Column": skewness.index,
    "Skewness": skewness.values
})

skewness_df
```

Skewness helps identify whether numerical distributions are approximately symmetric or strongly skewed.

### Numerical Feature Distributions

```python
# Plot distributions of numerical features

for column in numerical_columns:

    plt.figure(figsize=(8, 4))

    sns.histplot(
        data=df,
        x=column,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()
```

### Numerical Feature Box Plots

```python
# Generate box plots for numerical features

for column in numerical_columns:

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        data=df,
        x=column
    )

    plt.title(f"Box Plot of {column}")
    plt.tight_layout()
    plt.show()
```

Box plots provide a visual indication of the spread of numerical values and potential extreme observations.

### Potential Outlier Detection

The Interquartile Range (IQR) method is used to identify potential outliers.

```python
# Detect potential outliers using the IQR method

outlier_summary = []

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ).sum()

    outlier_summary.append({
        "Column": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Potential Outliers": outliers
    })

outlier_df = pd.DataFrame(outlier_summary)

outlier_df
```

The detected values are considered **potential outliers** and are not automatically removed.

---

## 6. Target Variable Analysis

The target variable for the healthcare disease prediction project is:

**Medical Condition**

The distribution of the target classes is examined to determine whether the classes are balanced.

### Target Variable Distribution

```python
# Check target variable distribution

target_column = "Medical Condition"

print("Target Variable:", target_column)
print("\nTarget Classes:\n")

print(df[target_column].value_counts())
```

### Target Class Percentages

```python
# Calculate target class percentages

target_distribution = pd.DataFrame({
    "Count": df[target_column].value_counts(),
    "Percentage": (
        df[target_column]
        .value_counts(normalize=True) * 100
    )
})

target_distribution
```

This shows the number and percentage of records belonging to each disease class.

### Target Distribution Plot

```python
# Plot target variable distribution

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x=target_column,
    order=df[target_column].value_counts().index
)

plt.title("Distribution of Medical Conditions")
plt.xlabel("Medical Condition")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## 7. Feature vs Target Analysis

Categorical and numerical features can be compared with the target variable to understand possible differences between disease classes.

### Categorical Features vs Target

```python
# Compare low-cardinality categorical features with the target

for column in low_cardinality_columns:

    if column != target_column:

        cross_tab = pd.crosstab(
            df[column],
            df[target_column],
            normalize="index"
        ) * 100

        print(f"\n{column} vs {target_column}")
        display(cross_tab.round(2))
```

This produces percentage-based cross-tabulations between categorical features and the target variable.

### Numerical Features vs Target

```python
# Compare numerical features across target classes

for column in numerical_columns:

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x=target_column,
        y=column
    )

    plt.title(f"{column} by Medical Condition")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

These box plots allow numerical feature distributions to be compared across medical-condition classes.

---

## 8. Feature Analysis Summary

Categorical features are examined using:

* Unique-value counts
* Frequency distributions
* Cardinality
* Low-cardinality identification
* Distribution plots
* Feature-versus-target comparisons

Numerical features are examined using:

* Descriptive statistics
* Distribution plots
* Box plots
* Skewness
* Potential outlier detection
* Numerical feature versus target comparisons

The analysis does not permanently remove or transform any feature.

---

## 9. Automatic Feature Analysis Summary

```python
# Generate automatic feature analysis summary

print("FEATURE ANALYSIS SUMMARY")
print("=" * 50)

print(f"Total records           : {len(df)}")
print(f"Categorical/Text columns: {len(categorical_columns)}")
print(f"Numerical columns       : {len(numerical_columns)}")
print(
    f"Low-cardinality categorical columns: "
    f"{len(low_cardinality_columns)}"
)

print("\nCategorical columns:")

for column in categorical_columns:
    print(
        f"- {column}: "
        f"{df[column].nunique(dropna=True)} unique values"
    )

print("\nNumerical columns:")

for column in numerical_columns:
    print(f"- {column}")

print("\nTarget variable:")
print(f"- {target_column}")
print(
    f"- Number of classes: "
    f"{df[target_column].nunique(dropna=True)}"
)
```

---

## 10. Data Quality and Feature Considerations

The analysis provides several points that should be considered during later preprocessing:

* Categorical variables may need encoding.
* High-cardinality text columns may require feature-selection consideration.
* Numerical variables may have different scales.
* Numerical distributions should be reviewed for skewness.
* Potential outliers should be investigated before deciding whether treatment is necessary.
* The target variable should be checked for class imbalance.
* Feature-target relationships should be evaluated without assuming that a relationship implies causation.



---

## 11. Conclusion

The categorical and numerical features of the healthcare dataset were analyzed to understand their structure, cardinality, distributions, descriptive statistics, skewness, and potential outliers.

The target variable, **Medical Condition**, was also analyzed to understand the distribution of disease classes.

The findings from this notebook will support the next stages of the project:

1. Exploratory Data Analysis
2. Initial data cleaning
3. Feature selection
4. Feature engineering
5. Machine learning preprocessing


