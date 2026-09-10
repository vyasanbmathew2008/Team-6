# 5) Initial Data Preprocessing & Dataset Documentation

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">](https://colab.research.google.com/github/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/notebooks/Week_1/05_Initial_Data_Preprocessing_Dataset_Documentation.ipynb)

## Notebook

[Open Notebook](../../notebooks/Week_1/05_Initial_Data_Preprocessing_Dataset_Documentation.ipynb)

## Dataset

[Healthcare Dataset](../../dataset/healthcare_dataset.csv)

---

## 1. Objective

This notebook performs initial data preprocessing and documents the characteristics of the healthcare disease prediction dataset.

The main preprocessing tasks include:

- Dataset inspection
- Duplicate record removal
- Missing value handling
- Text cleaning and formatting
- Date conversion
- Initial removal of low-value/high-cardinality columns
- Target variable validation
- Data-quality verification

The preprocessing is performed on a working DataFrame for further analysis and modelling.

---

## 2. Dataset Loading

The original healthcare dataset is loaded directly from the project's GitHub repository.

```python
url = "https://raw.githubusercontent.com/vyasanbmathew2008/Team-6/main/healthcare-disease-prediction/dataset/healthcare_dataset.csv"

df = pd.read_csv(url)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
````

The dataset contains information about patients, admissions, medical conditions, insurance, medications, test results, and other healthcare-related attributes.

---

## 3. Initial Dataset Inspection

The dataset is inspected using:

* `head()`
* `tail()`
* `shape`
* `columns`
* `info()`
* Data-type analysis

Example:

```python
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

df.info()
```

The dataset contains both numerical and categorical/text variables.

---

## 4. Working DataFrame

A copy of the original dataset is created for preprocessing.

```python
cleaned_df = df.copy()

print("Working DataFrame created.")
print("Shape:", cleaned_df.shape)
```

All preprocessing operations are performed on this working DataFrame.

---

## 5. Duplicate Record Preprocessing

Exact duplicate records are identified before being removed.

```python
duplicate_count = cleaned_df.duplicated().sum()

print("Duplicate rows before preprocessing:", duplicate_count)

duplicate_percentage = (
    duplicate_count / len(cleaned_df)
) * 100

print(f"Duplicate percentage: {duplicate_percentage:.2f}%")
```

Exact duplicate records are then removed:

```python
if duplicate_count > 0:
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

print(
    "Duplicate rows after preprocessing:",
    cleaned_df.duplicated().sum()
)
```

Removing exact duplicate records prevents the same observation from being unintentionally represented multiple times.

---

## 6. Missing Value Preprocessing

Missing values are checked before preprocessing.

```python
missing_before = cleaned_df.isnull().sum()

missing_summary = pd.DataFrame({
    "Missing Count": missing_before,
    "Missing Percentage":
        (missing_before / len(cleaned_df)) * 100
})

missing_summary = missing_summary[
    missing_summary["Missing Count"] > 0
]

display(missing_summary)
```

### Target Variable Check

The target variable is:

```python
target_column = "Medical Condition"
```

Missing target values are checked separately:

```python
print(
    "Missing target values:",
    cleaned_df[target_column].isnull().sum()
)
```

If missing target values exist, those rows are removed because supervised learning requires a known target.

```python
if target_column in cleaned_df.columns:
    cleaned_df = cleaned_df.dropna(
        subset=[target_column]
    ).reset_index(drop=True)
```

### Numerical Missing Values

Missing numerical values are handled using the median.

```python
numerical_columns = cleaned_df.select_dtypes(
    include=np.number
).columns.tolist()

for column in numerical_columns:
    if cleaned_df[column].isnull().sum() > 0:
        cleaned_df[column] = cleaned_df[column].fillna(
            cleaned_df[column].median()
        )
```

The median is used because it is less sensitive to extreme values than the mean.

### Categorical Missing Values

Missing categorical values are handled using the mode.

```python
categorical_columns = cleaned_df.select_dtypes(
    include="object"
).columns.tolist()

for column in categorical_columns:
    if cleaned_df[column].isnull().sum() > 0:
        mode_value = cleaned_df[column].mode()

        if not mode_value.empty:
            cleaned_df[column] = cleaned_df[column].fillna(
                mode_value.iloc[0]
            )
```

---

## 7. Remaining Missing Values

After preprocessing, the dataset is checked again.

```python
remaining_missing = cleaned_df.isnull().sum()

remaining_missing = remaining_missing[
    remaining_missing > 0
]

if remaining_missing.empty:
    print("No missing values remain in the working DataFrame.")
else:
    display(remaining_missing)
```

This verifies whether the missing-value preprocessing was successful.

---

## 8. Text Cleaning and Formatting

Text columns are cleaned by removing unnecessary leading and trailing whitespace.

```python
text_columns = cleaned_df.select_dtypes(
    include="object"
).columns.tolist()

for column in text_columns:
    cleaned_df[column] = cleaned_df[column].str.strip()
```

### Name Capitalization

The `Name` column contains inconsistent capitalization in some records.

For example:

* `Bobby JacksOn`
* `LesLie TErRy`
* `DaNnY sMitH`

The names are standardized using title case:

```python
if "Name" in cleaned_df.columns:
    cleaned_df["Name"] = cleaned_df["Name"].str.title()
```

This improves formatting consistency without changing the underlying patient names.

---

## 9. Categorical Value Inspection

Categorical values are inspected after text cleaning.

```python
categorical_columns = cleaned_df.select_dtypes(
    include="object"
).columns.tolist()

for column in categorical_columns:
    print(f"\n--- {column} ---")
    print(
        cleaned_df[column]
        .value_counts(dropna=False)
        .head(20)
    )
```

This helps identify unexpected or inconsistent category values.

---

## 10. Date Preprocessing

The dataset contains date-related variables such as:

* `Date of Admission`
* `Discharge Date`

The date columns are identified automatically:

```python
date_columns = [
    column
    for column in cleaned_df.columns
    if "date" in column.lower()
]

print("Date-related columns:")
print(date_columns)
```

The identified columns are converted to pandas datetime format:

```python
for column in date_columns:
    cleaned_df[column] = pd.to_datetime(
        cleaned_df[column],
        errors="coerce"
    )
```

The conversion allows the date information to be correctly interpreted during later analysis and feature engineering.

---

## 11. Initial Irrelevant / Low-Value Column Preprocessing

Based on the previous dataset analysis, some columns are initially considered low-value or unsuitable for general disease prediction.

The following columns are selected for removal:

```python
columns_to_remove = [
    "Name",
    "Doctor",
    "Hospital",
    "Room Number"
]
```

These columns may represent patient identifiers, individual healthcare personnel, specific institutions, or room information rather than generalizable patient characteristics.

The columns are removed from the working DataFrame:

```python
existing_columns_to_remove = [
    column
    for column in columns_to_remove
    if column in cleaned_df.columns
]

cleaned_df = cleaned_df.drop(
    columns=existing_columns_to_remove
)
```

The remaining columns are then inspected:

```python
print("Columns remaining after preprocessing:")
print(cleaned_df.columns.tolist())

print("Current shape:", cleaned_df.shape)
```

---

## 12. Prediction-Timing Considerations

Some variables may contain information that is only available after or during treatment.

Examples include:

* `Discharge Date`
* `Medication`
* `Test Results`
* `Billing Amount`

These variables are not automatically removed during the initial preprocessing because their usefulness depends on the exact prediction task and when the prediction is expected to be made.

Potential data leakage should therefore be considered before model training.

---

## 13. Derived Length of Stay Check

Length of stay can be calculated using the admission and discharge dates.

```python
if (
    "Date of Admission" in cleaned_df.columns
    and "Discharge Date" in cleaned_df.columns
):

    cleaned_df["Length of Stay"] = (
        cleaned_df["Discharge Date"]
        - cleaned_df["Date of Admission"]
    ).dt.days

    print("Length of Stay calculated successfully.")
```

The derived variable can be inspected using:

```python
cleaned_df["Length of Stay"].describe()
```

This calculation is used as a preprocessing and analysis check. Its final use as a modelling feature should be decided according to the prediction scenario.

---

## 14. Target Variable Validation

The target variable for the disease prediction task is:

**`Medical Condition`**

The target categories are inspected:

```python
print("Target variable:", target_column)

print("\nTarget categories:")
print(cleaned_df[target_column].unique())

print(
    "\nNumber of target classes:",
    cleaned_df[target_column].nunique()
)
```

The target distribution is also calculated:

```python
target_distribution = (
    cleaned_df[target_column]
    .value_counts()
    .to_frame("Count")
)

target_distribution["Percentage"] = (
    target_distribution["Count"]
    / len(cleaned_df)
) * 100

display(target_distribution)
```

This helps determine whether the target classes are reasonably distributed or whether class imbalance may need to be considered during modelling.

---

## 15. Target Distribution Visualization

The distribution of the medical conditions can be visualized using a count plot.

```python
plt.figure(figsize=(10, 5))

sns.countplot(
    data=cleaned_df,
    x=target_column
)

plt.title(
    "Medical Condition Distribution After Preprocessing"
)

plt.xlabel("Medical Condition")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

---

## 16. Final Data-Quality Checks

The working DataFrame is checked after preprocessing.

```python
print(
    "Final working DataFrame shape:",
    cleaned_df.shape
)

print(
    "\nRemaining duplicate rows:",
    cleaned_df.duplicated().sum()
)

print(
    "\nTotal remaining missing values:",
    cleaned_df.isnull().sum().sum()
)
```

The final data types are also inspected:

```python
display(
    pd.DataFrame({
        "Column": cleaned_df.columns,
        "Data Type":
            cleaned_df.dtypes.astype(str)
    })
)
```

---

## 17. Dataset Characteristics After Initial Preprocessing

The dataset is documented in terms of:

* Number of records
* Number of features
* Numerical variables
* Categorical variables
* Date variables
* Target variable
* Duplicate records
* Missing values
* Initially removed columns

Example:

```python
print("===== DATASET CHARACTERISTICS =====")

print("Number of rows:", cleaned_df.shape[0])
print("Number of columns:", cleaned_df.shape[1])

print("\nNumerical columns:")
print(
    cleaned_df.select_dtypes(
        include=np.number
    ).columns.tolist()
)

print("\nCategorical/Text columns:")
print(
    cleaned_df.select_dtypes(
        include="object"
    ).columns.tolist()
)

print("\nDate columns:")
print(
    cleaned_df.select_dtypes(
        include="datetime"
    ).columns.tolist()
)

print("\nTarget variable:", target_column)

print(
    "Number of target classes:",
    cleaned_df[target_column].nunique()
)

print(
    "\nDuplicate rows:",
    cleaned_df.duplicated().sum()
)

print(
    "\nTotal missing values:",
    cleaned_df.isnull().sum().sum()
)
```

---

## 18. Preprocessing Summary

The following initial preprocessing methods were applied:

1. Created a working copy of the original dataset.
2. Inspected dataset structure and data types.
3. Identified exact duplicate records.
4. Removed exact duplicate records.
5. Checked missing values.
6. Removed rows with missing target values when necessary.
7. Filled missing numerical values using the median.
8. Filled missing categorical values using the mode.
9. Removed unnecessary leading and trailing whitespace.
10. Standardized name capitalization using title case.
11. Inspected categorical values for consistency.
12. Converted date columns to datetime format.
13. Initially removed low-value/high-cardinality columns.
14. Considered potential prediction-time data leakage.
15. Inspected the derived length of stay.
16. Validated the target variable.
17. Checked target class distribution.
18. Performed final data-quality checks.

---

## 19. Important Data-Quality Considerations

### Prediction Timing

Variables such as medication, test results, billing amount, and discharge date should be evaluated according to when the prediction is intended to occur.

### Feature Selection

The usefulness of the remaining features should be confirmed during the modelling stage.

### Date Features

Raw dates may later be transformed into useful features such as:

* Admission year
* Admission month
* Admission day
* Length of stay

### Categorical Encoding

Categorical variables will need to be encoded before being supplied to most machine-learning algorithms.

### Numerical Scaling

Scaling may be required for algorithms that are sensitive to differences in feature magnitude.

### Data Leakage

Features that contain information unavailable at prediction time should be carefully evaluated to avoid data leakage.

---

## 20. Conclusion

The healthcare disease prediction dataset was inspected and initially preprocessed.

The main preprocessing tasks included:

* Duplicate removal
* Missing-value handling
* Text standardization
* Date conversion
* Initial removal of low-value/high-cardinality columns
* Target validation
* Data-quality verification

The target variable `Medical Condition` was also validated and its class distribution was inspected.

The resulting working DataFrame is ready for the next stages of feature preparation and machine-learning modelling.


