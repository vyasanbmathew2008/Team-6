# 4) Exploratory Data Analysis (EDA)

[<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">](https://colab.research.google.com/github/vyasanbmathew2008/Team-6/blob/main/healthcare-disease-prediction/notebooks/Week_1/04_Exploratory_Data_Analysis.ipynb)

## Notebook

[04_Exploratory_Data_Analysis.ipynb](../../notebooks/Week_1/04_Exploratory_Data_Analysis.ipynb)

## Dataset

[healthcare_dataset.csv](../../dataset/healthcare_dataset.csv)

---

## Objective

The objective of this notebook is to perform **Exploratory Data Analysis (EDA)** on the healthcare disease prediction dataset.

The analysis focuses on understanding:

- Dataset structure
- Missing values
- Target variable distribution
- Numerical feature distributions
- Categorical feature distributions
- Relationships between features and medical conditions
- Correlations between numerical variables
- Potential outliers
- Important data-quality considerations

No permanent data cleaning or feature transformation is performed in this notebook.

---

## 1. Dataset Overview

The dataset is loaded directly from the GitHub repository.

The following aspects are examined:

- Number of records
- Number of columns
- Column names
- Data types
- First few records
- Descriptive statistics

This provides an initial understanding of the dataset structure.

---

## 2. Missing Value Analysis

Missing values are examined using:

- Missing value counts
- Missing value percentages
- Missing-value heatmap

This helps identify columns that may require missing-value treatment during the data-cleaning stage.

No missing values are permanently filled or removed in this notebook.

---

## 3. Target Variable Analysis

The target variable for the project is:

**Medical Condition**

The distribution of the target classes is analyzed using:

- Frequency counts
- Class percentages
- Count plots

This helps determine whether the disease classes are approximately balanced or whether class imbalance may need to be considered during model development.

---

## 4. Numerical Feature Analysis

Numerical features are analyzed using:

- Descriptive statistics
- Histograms
- Kernel density estimates
- Box plots
- Outlier inspection

The main numerical features include variables such as:

- Age
- Billing Amount
- Room Number

The distributions are examined to identify unusual values, skewness, and potential outliers.

---

## 5. Age Analysis

Age is investigated as an important numerical feature.

The analysis includes:

- Overall age distribution
- Age distribution across medical conditions
- Box plots comparing age between disease classes

This helps determine whether different medical conditions show different age distributions.

---

## 6. Billing Amount Analysis

Billing Amount is examined as a numerical feature.

The analysis includes:

- Billing amount distribution
- Billing amount across medical conditions
- Box plots for identifying potential differences and outliers

Billing Amount should also be reviewed later for possible data leakage depending on when the prediction is intended to be made.

---

## 7. Categorical Feature Analysis

Categorical features are analyzed using:

- Frequency counts
- Count plots
- Cross-tabulations
- Feature-versus-target comparisons

Important categorical features include:

- Gender
- Blood Type
- Admission Type
- Insurance Provider
- Medication
- Test Results
- Medical Condition

High-cardinality categorical columns such as Doctor and Hospital should be evaluated carefully before machine learning preprocessing.

---

## 8. Gender Analysis

The distribution of Gender is examined using a count plot.

Gender is also compared with Medical Condition using a cross-tabulation.

This helps identify whether disease distributions differ across gender categories.

---

## 9. Admission Type Analysis

Admission Type is analyzed to understand the distribution of admission categories.

The categories are also compared with Medical Condition.

This can help identify whether certain medical conditions occur more frequently under particular admission types.

---

## 10. Blood Type Analysis

Blood Type is analyzed using:

- Frequency distribution
- Count plot
- Comparison with Medical Condition

This provides an initial understanding of whether blood-type categories show differences across disease classes.

---

## 11. Insurance Provider Analysis

Insurance Provider is examined using:

- Frequency counts
- Count plots
- Comparison with Medical Condition

The results can be used to determine whether insurance categories have meaningful differences in the dataset.

---

## 12. Correlation Analysis

Correlation analysis is performed on numerical variables.

A correlation matrix and heatmap are used to identify relationships between numerical features.

Strong correlations should be investigated further during feature engineering and model development.

---

## 13. Numerical Features vs Target

Numerical features are compared across Medical Condition categories using box plots.

This allows us to visually examine whether variables such as:

- Age
- Billing Amount
- Room Number

have different distributions across disease classes.

---

## 14. Medication Analysis

Medication is a categorical feature that may contain useful information about disease patterns.

The analysis includes:

- Medication frequency distribution
- Medication versus Medical Condition comparison

However, Medication should be evaluated based on the intended prediction timing because medication information may not be available before a prediction is made.

---

## 15. Test Results Analysis

Test Results are analyzed using:

- Frequency distribution
- Count plot
- Comparison with Medical Condition

Test Results may contain useful predictive information.

However, whether this feature should be used for prediction depends on when the prediction is intended to occur.

---

## 16. Date Feature Analysis

The dataset contains two date-related columns:

- Date of Admission
- Discharge Date

These columns are initially inspected as text values.

During later feature engineering, they may be converted into datetime values and used to derive features such as:

- Admission year
- Admission month
- Admission day
- Length of stay

No permanent date transformation is performed in this EDA notebook.

---

## 17. Data Patterns and Observations

The following aspects should be considered based on the exploratory analysis:

- Distribution of medical condition classes
- Age distribution
- Billing amount distribution
- Categorical feature distributions
- Feature-versus-target relationships
- Potential numerical outliers
- Correlations between numerical variables
- High-cardinality categorical features
- Potential data leakage
- Prediction timing

These observations will guide the next stages of the project.

---

## 18. Important Data Quality Considerations

Several data-quality considerations were identified during EDA:

1. Duplicate records should be investigated before model training.
2. Missing values should be handled during the cleaning/preprocessing stage.
3. High-cardinality columns such as Doctor and Hospital may require special treatment.
4. Name and Room Number may have limited predictive value.
5. Date columns may require feature extraction.
6. Medication and Test Results should be evaluated according to prediction timing.
7. Discharge Date may introduce data leakage if the prediction is intended to occur at admission.
8. Billing Amount should be reviewed for possible leakage depending on the prediction scenario.
9. Numerical outliers should be investigated before deciding whether treatment is required.
10. Categorical variables will require appropriate encoding before machine learning.

---

## 19. EDA Summary

The healthcare dataset was explored through:

- Dataset structure analysis
- Missing value analysis
- Target variable distribution
- Numerical feature distributions
- Categorical feature distributions
- Age analysis
- Billing amount analysis
- Correlation analysis
- Feature-versus-target comparisons
- Date feature inspection
- Data-quality assessment

The analysis provides a foundation for:

- Data cleaning
- Feature selection
- Feature engineering
- Encoding
- Scaling
- Machine learning model development

---

## Conclusion

Exploratory Data Analysis was performed to understand the characteristics, distributions, relationships, and potential issues within the healthcare disease prediction dataset.

The findings from this analysis will be used to guide the subsequent stages of:

- Initial data cleaning
- Feature engineering
- Data preprocessing
- Feature selection
- Model development
- Model evaluation

No permanent changes were made to the original dataset during this notebook.
