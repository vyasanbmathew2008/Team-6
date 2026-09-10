# Healthcare Disease Prediction — Data Types & Missing Values

## Notebook

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">](https://colab.research.google.com/github/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/notebooks/Week_1/01_Data_Types_Missing_Values.ipynb)

**Notebook:** [`01_Data_Types_Missing_Values.ipynb`](../../notebooks/Week_1/01_Data_Types_Missing_Values.ipynb)

---

# 1. Data Types & Missing Values

## Objective

The objective of this analysis is to understand the structure and quality of the healthcare dataset before applying machine learning preprocessing techniques.

The analysis covers:

* Dataset structure
* Number of records and columns
* Column names
* Data types
* Numerical columns
* Categorical/text columns
* Date-related columns
* Unique values
* Missing values
* Missing-value percentages
* Overall dataset summary

No permanent data-cleaning operations are performed in this notebook.

---

# 2. Import Required Libraries

The following Python libraries are used for data loading, analysis, and visualization.

### Colab Cell 2

```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

---

# 3. Load Dataset

The healthcare dataset is loaded directly from the project's GitHub repository.

### Colab Cell 3

```python
# Load dataset directly from GitHub
url = "https://raw.githubusercontent.com/vyasanbmathew2008/Team-6/main/healthcare-disease-prediction/dataset/healthcare_dataset.csv"
df = pd.read_csv(url)
print("Dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")
```

The dataset used in this analysis is:

[`healthcare_dataset.csv`](../../dataset/healthcare_dataset.csv)

---

# 4. Display Basic Dataset Information

### Colab Cell 4

```python
# Display basic dataset information
print("Dataset Shape:", df.shape)
```

This displays the number of rows and columns in the dataset.

---

# 5. Display First 5 Records

### Colab Cell 5

```python
# Display first 5 records
df.head()
```

The `head()` function is used to inspect the first five records of the dataset.

---

# 6. Display Last 5 Records

### Colab Cell 6

```python
# Display last 5 records
df.tail()
```

The `tail()` function is used to inspect the last five records of the dataset.

---

# 7. Get Number of Rows and Columns

### Colab Cell 7

```python
# Get number of rows and columns
rows, columns = df.shape
print(f"Number of rows    : {rows}")
print(f"Number of columns : {columns}")
```

This provides the exact number of records and features present in the dataset.

---

# 8. Display Column Names

### Colab Cell 8

```python
# Display all column names
print("Columns in the dataset:\n")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")
```

This lists every column in the dataset with its corresponding number.

---

# 9. Dataset Information

### Colab Cell 9

```python
# Display dataset information
df.info()
```

The `info()` function provides information about:

* Column names
* Number of non-null values
* Data types
* Memory usage

---

# 10. Data Type of Every Column

### Colab Cell 10

```python
# Display data type of every column
dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values
})
dtype_df
```

This creates a table showing the data type of every dataset column.

---

# 11. Count Columns by Data Type

### Colab Cell 11

```python
# Count columns by data type
print("Number of columns by data type:\n")
print(df.dtypes.value_counts())
```

This identifies how many columns belong to each data type.

---

# 12. Numerical Columns

Numerical columns contain values represented by numbers and may be used directly or transformed during machine learning preprocessing.

### Colab Cell 12

```python
# Identify numerical columns
numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
print("Numerical Columns:\n")
for column in numerical_columns:
    print("-", column)
print(f"\nTotal numerical columns: {len(numerical_columns)}")
```

This identifies all numerical columns and counts them.

---

# 13. Categorical/Text Columns

Categorical or text columns contain labels, categories, names, or other textual information.

These columns may require encoding before being used by machine learning algorithms.

### Colab Cell 13

```text
##Categorical/Text Columns

Categorical or text columns contain labels, categories, names, or other textual information.
These columns may require encoding before being used by machine learning algorithms.
```

### Colab Cell 14

```python
# Identify categorical/text columns
categorical_columns = df.select_dtypes(include="object").columns.tolist()
print("Categorical/Text Columns:\n")
for column in categorical_columns:
    print("-", column)
print(f"\nTotal categorical/text columns: {len(categorical_columns)}")
```

This identifies the categorical and text-based columns in the dataset.

---

# 14. Date-Related Columns

Columns containing dates need special consideration because they may initially be loaded as text/object data.

### Colab Cell 15

```python
# Identify columns containing "date" in their name
date_columns = [
    column for column in df.columns
    if "date" in column.lower()
]
print("Potential Date Columns:\n")
for column in date_columns:
    print("-", column)
```

This searches for columns whose names contain the word `date`.

### Colab Cell 16

```python
# Display data types of date-related columns
print("Date Column Data Types:\n")
for column in date_columns:
    print(f"{column}: {df[column].dtype}")
```

This checks the current data type of the identified date-related columns.

---

# 15. Unique Values

The number of unique values in each column is examined to understand the characteristics of the dataset.

### Colab Cell 17

```text
##Unique Values
```

### Colab Cell 18

```python
# Count unique values in each column
unique_values_df = pd.DataFrame({
    "Column": df.columns,
    "Unique Values": [
        df[column].nunique(dropna=True)
        for column in df.columns
    ]
})
unique_values_df
```

Unique-value counts can help identify:

* Low-cardinality categorical features
* High-cardinality features
* Potential identifier-like columns
* Variables that may require additional preprocessing

A high number of unique values does not automatically mean that a column should be removed.

---

# 16. Complete Column Summary

A combined summary is created containing data type, non-null count, missing count, and unique values.

### Colab Cell 19

```python
# Create complete column summary
column_summary = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Non-Null Count": df.notnull().sum().values,
    "Missing Count": df.isnull().sum().values,
    "Unique Values": df.nunique(dropna=True).values
})
column_summary
```

This table provides a consolidated view of the dataset's column-level characteristics.

---

# 17. Missing Value Analysis

Missing values can affect data analysis and machine learning models.

The following are checked:

1. Missing values in each column
2. Missing-value percentage
3. Columns containing missing values
4. Total missing values in the dataset

### Colab Cell 20

```text
# Missing Value Analysis

Missing values can affect data analysis and machine learning models.
We will check:
1. Missing values in each column
2. Missing-value percentage
3. Columns containing missing values
4. Total missing values in the dataset
```

---

# 18. Count Missing Values

### Colab Cell 21

```python
# Count missing values in each column
missing_count = df.isnull().sum()
print("Missing values in each column:\n")
print(missing_count)
```

This calculates the number of missing values in every column.

---

# 19. Calculate Missing-Value Percentage

### Colab Cell 22

```python
# Calculate missing-value percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum().values,
    "Missing Percentage": missing_percentage.values
})
missing_df = missing_df.sort_values(
    by="Missing Values",
    ascending=False
)
missing_df
```

The missing-value percentage allows the severity of missing data to be compared across columns.

---

# 20. Display Columns Containing Missing Values

### Colab Cell 23

```python
# Display only columns containing missing values
missing_only = missing_df[
    missing_df["Missing Values"] > 0
]
if len(missing_only) > 0:
    display(missing_only)
else:
    print("No columns contain missing values.")
```

This displays only the columns that actually contain missing values.

---

# 21. Calculate Overall Missing Values

The total number of missing cells is calculated before determining the overall missing percentage.

### Colab Cell 24

```python
# Calculate overall percentage of missing cells
total_missing = df.isnull().sum().sum()

total_cells = df.shape[0] * df.shape[1]
overall_missing_percentage = (
    total_missing / total_cells
) * 100

print(
    f"Overall percentage of missing cells: "
    f"{overall_missing_percentage:.2f}%"
)
```

The `total_missing` calculation is included here because it is required by this cell and the dataset summary in Cell 26.

---

# 22. Dataset Summary

The final section generates an automatic summary of the dataset characteristics.

### Colab Cell 25

```text
#Dataset Summary
```

### Colab Cell 26

```python
# Generate automatic summary values for documentation
print("DATASET SUMMARY")
print("=" * 50)
print(f"Number of records       : {df.shape[0]}")
print(f"Number of columns       : {df.shape[1]}")
print(f"Numerical columns       : {len(numerical_columns)}")
print(f"Categorical/Text columns: {len(categorical_columns)}")
print(f"Date-related columns   : {len(date_columns)}")
print(f"Total missing values    : {total_missing}")
print(
    f"Missing cell percentage : "
    f"{overall_missing_percentage:.2f}%"
)

if total_missing > 0:
    highest_missing_column = missing_df.iloc[0]["Column"]
    highest_missing_count = missing_df.iloc[0]["Missing Values"]
    print(
        f"Highest missing column  : "
        f"{highest_missing_column} "
        f"({highest_missing_count} values)"
    )
else:
    print("Highest missing column  : None")
```

This automatically summarizes:

* Number of records
* Number of columns
* Number of numerical columns
* Number of categorical/text columns
* Number of date-related columns
* Total missing values
* Overall missing-cell percentage
* Column with the highest number of missing values

---

# 23. Data Quality Observations

Based on the outputs generated in Google Colab, the following characteristics should be documented.

### Dataset Structure

The notebook identifies:

* Total number of records
* Total number of columns
* Numerical features
* Categorical/text features
* Date-related features

### Data Types

The analysis checks whether each feature is numerical, categorical/text, or date-related.

The date-related columns include:

* `Date of Admission`
* `Discharge Date`

These may initially be stored as `object` values when loaded from the CSV and can be converted to datetime during later preprocessing.

### Unique Values

Unique-value counts are used to understand the cardinality of each feature.

Columns with many unique values may require additional feature-selection consideration during later stages.

### Missing Values

The notebook calculates:

* Missing values per column
* Missing percentage per column
* Columns containing missing values
* Total missing values
* Overall missing-cell percentage

---


# 24. Conclusion

The healthcare dataset was inspected to understand its structure, data types, unique values, and missing-value characteristics.

The results from this notebook provide the foundation for the next stages of the project, including:

1. Duplicate and irrelevant-column analysis
2. Categorical and numerical feature analysis
3. Exploratory Data Analysis
4. Initial data cleaning
5. Machine learning preprocessing

---

