# Salary Predictor & Classifier

A full-stack ML web application that predicts data science salaries and classifies them as above/below median using multiple machine learning models.

## Features

- **Salary Prediction** — Linear Regression model predicts salary in USD based on job attributes
- **Salary Classification** — KNN and SVM classifiers determine if a salary is above or below the median
- **Interactive Frontend** — Modern, responsive UI with real-time predictions
- **REST API** — Flask backend with endpoints for predictions, options, and model scores

## Tech Stack

- **Backend**: Python, Flask, scikit-learn, pandas, NumPy
- **Frontend**: HTML, CSS, JavaScript
- **ML Models**: Linear Regression, K-Nearest Neighbors, Support Vector Machine
- **Data Processing**: PCA, StandardScaler, IsolationForest (outlier removal)

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Open in browser**: Navigate to `http://localhost:5000`

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the frontend |
| `/api/options` | GET | Returns dropdown options for form fields |
| `/api/scores` | GET | Returns model performance scores |
| `/api/predict` | POST | Accepts job attributes, returns salary prediction & classification |

## Dataset

Uses the [Data Science Salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries) dataset (`ds_salaries.csv`).
