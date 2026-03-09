# Steel Structural Strength Prediction using Machine Learning

## Overview

This project develops a **Machine Learning model to predict the structural failure load of steel components** based on material properties, geometry, and environmental conditions.
The system applies data preprocessing, feature engineering, model comparison, and hyperparameter tuning to determine the most effective algorithm for predicting steel strength.

The project also includes a **Streamlit web application** that allows users to input steel parameters and obtain predicted structural load capacity.

---

## Problem Statement

Structural engineers need reliable methods to estimate the **maximum load that steel components can withstand before failure**.
This project uses machine learning to analyze historical steel property data and predict structural performance.

---

## Dataset Features

The dataset contains the following parameters:

* **Steel Grade** – Type/grade of steel material
* **Yield Strength** – Maximum stress before permanent deformation
* **Cross-Sectional Area** – Area of the steel member
* **Applied Load** – Load acting on the structure
* **Length** – Length of the steel component
* **Temperature Conditions** – Environmental temperature affecting steel strength

Additional engineered features include:

* **Stress**
* **Slenderness Ratio**
* **Normalized Load Features**

---

## Project Pipeline

### 1. Data Processing

* Remove duplicate records
* Handle missing values
* Encode categorical variables
* Normalize numerical features

### 2. Feature Engineering

Structural features are derived to improve model performance:

* Stress calculation
* Slenderness ratio
* Load distribution features

### 3. Model Training

Multiple regression algorithms are trained and compared:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* Support Vector Regressor

Hyperparameter tuning is applied to select the best performing model.

### 4. Model Evaluation

Models are evaluated using:

* **R² Score**
* **RMSE (Root Mean Squared Error)**

The best model is saved as:

```
models/best_model.pkl
```

---

## Project Structure

```
ML_Steel_Project/
│
├── train_pipeline.py
├── requirements.txt
├── README.md
│
├── data/
│   └── steel_dataset.csv
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluate_model.py
│
└── app/
    └── streamlit_app.py
```

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train_pipeline.py
```

### Launch Web Application

```bash
streamlit run app/streamlit_app.py
```

---

## Output

The system predicts:

**Estimated structural failure load of steel components**

This helps engineers understand **maximum load capacity before structural failure occurs**.

---

## Technologies Used

* Python
* Scikit-Learn
* Pandas
* NumPy
* Streamlit
* Machine Learning Regression Models

---

## Future Improvements

* Deep learning based structural prediction
* Larger structural engineering datasets
* Real-time engineering simulation integration
* Structural safety classification model

---

## Author

AI & Data Science Student | Machine Learning Enthusiast
