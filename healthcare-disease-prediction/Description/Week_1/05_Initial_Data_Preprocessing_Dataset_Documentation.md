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
