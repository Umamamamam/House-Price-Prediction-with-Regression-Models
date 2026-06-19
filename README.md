# 🏠 House Price Prediction

Predicting house prices with EDA, feature engineering, and 5 regression models — Gradient Boosting achieves ~90% R².

A machine learning project that predicts house sale prices based on property features (size, quality, location-related attributes, etc.) using the Ames Housing dataset. Multiple regression models are trained and compared to find the best performer.

## Libraries Used

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
- Jupyter Notebook

## Models Trained

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor (Best Model — R² ≈ 0.9006)
- XGBoost Regressor

**Project Info**

![License](https://img.shields.io/badge/License-MIT-lightgrey)
![R2 Score](https://img.shields.io/badge/Best%20R²-0.9006-success)

## Overview

This project covers the full ML workflow:

- **Data Cleaning** — handled missing values across 19+ columns using median/mode imputation and column drops (`Alley`, `MiscFeature`, `PoolQC` had too many missing values to be useful)
- **Outlier Treatment** — used the IQR method to clip outliers in key numeric features (`LotFrontage`, `GrLivArea`, `GarageArea`, `TotalBsmtSF`, etc.)
- **Exploratory Data Analysis (EDA)** — distribution plots, boxplots, scatterplots, and a correlation heatmap to understand relationships between features and `SalePrice`
- **Feature Engineering** — log-transformed `SalePrice` to reduce skew, one-hot encoded categorical variables, scaled features with `StandardScaler`
- **Modeling** — trained and compared 5 regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.0934 | 0.1597 | 0.8633 |
| Decision Tree | 0.1454 | 0.2015 | 0.7824 |
| Random Forest | 0.0991 | 0.1463 | 0.8853 |
| **Gradient Boosting** | **0.0940** | **0.1362** | **0.9006** |
| XGBoost | 0.1028 | 0.1452 | 0.8870 |

🏆 **Best Model: Gradient Boosting Regressor** — R² ≈ **0.9006** (~90%)

*(MAE/RMSE values are on log-transformed `SalePrice`.)*

The most influential features for predicting price were **Overall Quality, Living Area, Basement Area, Garage Capacity, and Year Built**.

## Key Takeaways

- Buyers on a budget may prefer smaller, older homes.
- Buyers wanting premium homes should look for high overall quality, larger living area, and modern facilities.
- Larger families benefit from more rooms, bathrooms, and basement space.
- Newer homes tend to have higher resale value, useful for investment buyers.

## Project Structure

```
house-price-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│   └── House_Price_Prediction.ipynb
└── data/
    └── data.csv          # not included — see Dataset section below
```

## Dataset

This project uses the **Ames Housing dataset** (commonly distributed via the Kaggle "House Prices - Advanced Regression Techniques" competition).

The raw data file is **not included** in this repo (per Kaggle's terms of use / to keep the repo lightweight). To run the notebook:

1. Download `data.csv` (train.csv) from [Kaggle's House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
2. Place it inside a `data/` folder at the project root
3. Update the path in the notebook's first cell (see note below)

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Umamamamam/house-price-prediction.git
cd house-price-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place `data.csv` inside a `data/` folder (see Dataset section above).

### 4. Run the notebook

```bash
jupyter notebook notebooks/House_Price_Prediction.ipynb
```

> **Note:** The first cell currently reads the file from `/content/data.csv` (a Google Colab path). Change it to a relative path before running locally:
> ```python
> data = pd.read_csv("data/data.csv")
> ```

## Tech Stack

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost


## Author

**Umamaheshwara Reddy**

- GitHub: [@Umamamamam](https://github.com/Umamamamam)
- LinkedIn: [umamaheshwara-reddy](https://www.linkedin.com/in/umamaheshwara-reddy-193194279/)
