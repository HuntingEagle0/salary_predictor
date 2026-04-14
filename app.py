"""
Salary Predictor & Classifier — Flask Backend
Replicates the ML pipeline from the SalaryPredictor notebook and exposes
REST API endpoints for predictions.
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ── Global state ────────────────────────────────────────────────────────
label_encoders: dict[str, LabelEncoder] = {}
scaler: StandardScaler | None = None
pca: PCA | None = None
lr_model: LinearRegression | None = None
knn_model: KNeighborsClassifier | None = None
svm_model: SVC | None = None
options: dict = {}
model_scores: dict = {}
median_salary: float = 0.0
cat_columns: list[str] = []

CSV_PATH = os.path.join(os.path.dirname(__file__), "ds_salaries.csv")


def train_models():
    """Replicate the notebook pipeline and train all three models."""
    global label_encoders, scaler, pca
    global lr_model, knn_model, svm_model
    global options, model_scores, median_salary, cat_columns

    df = pd.read_csv(CSV_PATH)
    df = df.dropna()

    # ── Capture raw option values BEFORE encoding ──────────────────
    cat_columns_local = df.select_dtypes(include="object").columns.tolist()
    cat_columns = cat_columns_local
    opts: dict = {}
    for col in cat_columns_local:
        opts[col] = sorted(df[col].unique().tolist())
    # Numeric options
    opts["work_year"] = sorted(df["work_year"].unique().tolist())
    opts["remote_ratio"] = sorted(df["remote_ratio"].unique().tolist())
    options = opts

    # ── Label-encode ───────────────────────────────────────────────
    label_encoders = {}
    for col in cat_columns_local:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # ── Features / targets ─────────────────────────────────────────
    X = df.drop("salary_in_usd", axis=1)
    y_reg = df["salary_in_usd"]
    median_salary = float(y_reg.median())
    y_clf = (df["salary_in_usd"] > median_salary).astype(int)

    # ── Scale ──────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Outlier removal ────────────────────────────────────────────
    iso = IsolationForest(contamination=0.05, random_state=42)
    outliers = iso.fit_predict(X_scaled)
    X_scaled = X_scaled[outliers == 1]
    y_reg = y_reg[outliers == 1]
    y_clf = y_clf[outliers == 1]

    # ── PCA ────────────────────────────────────────────────────────
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)

    # ── Regression ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_reg, test_size=0.2, random_state=42
    )
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    model_scores["regression_r2"] = round(lr_model.score(X_test, y_test), 4)

    # ── Classification ─────────────────────────────────────────────
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_pca, y_clf, test_size=0.2, random_state=42
    )

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_c, y_train_c)
    model_scores["knn_accuracy"] = round(
        accuracy_score(y_test_c, knn_model.predict(X_test_c)), 4
    )

    svm_model = SVC()
    svm_model.fit(X_train_c, y_train_c)
    model_scores["svm_accuracy"] = round(
        accuracy_score(y_test_c, svm_model.predict(X_test_c)), 4
    )

    print("[OK]  Models trained successfully!")
    print(f"   Linear Regression R²  : {model_scores['regression_r2']}")
    print(f"   KNN accuracy          : {model_scores['knn_accuracy']}")
    print(f"   SVM accuracy          : {model_scores['svm_accuracy']}")
    print(f"   Median salary (USD)   : {median_salary}")


# ── Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/options")
def get_options():
    """Return the distinct values for each dropdown field."""
    return jsonify(options)


@app.route("/api/scores")
def get_scores():
    """Return model performance scores."""
    return jsonify({**model_scores, "median_salary_usd": median_salary})


@app.route("/api/predict", methods=["POST"])
def predict():
    """Accept user input, run through the pipeline, return predictions."""
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON payload"}), 400

        required_fields = [
            "work_year", "experience_level", "employment_type",
            "job_title", "salary", "salary_currency",
            "employee_residence", "remote_ratio",
            "company_location", "company_size",
        ]
        for f in required_fields:
            if f not in data:
                return jsonify({"error": f"Missing field: {f}"}), 400

        # Build a single-row DataFrame matching the training columns
        row: dict = {}
        row["work_year"] = int(data["work_year"])
        row["salary"] = float(data["salary"])
        row["remote_ratio"] = int(data["remote_ratio"])

        # Encode categoricals
        for col in cat_columns:
            raw_val = data.get(col)
            le = label_encoders.get(col)
            if le is None:
                return jsonify({"error": f"Unknown column: {col}"}), 400
            if raw_val not in le.classes_:
                return jsonify({
                    "error": f"Unknown value '{raw_val}' for '{col}'. "
                             f"Valid: {le.classes_.tolist()}"
                }), 400
            row[col] = le.transform([raw_val])[0]

        # The training features included `salary` but NOT `salary_in_usd`.  
        # Column order must match the training set exactly.
        feature_order = [
            "work_year", "experience_level", "employment_type",
            "job_title", "salary", "salary_currency",
            "employee_residence", "remote_ratio",
            "company_location", "company_size",
        ]
        X_input = np.array([[row[c] for c in feature_order]])

        # Scale → PCA
        X_scaled = scaler.transform(X_input)
        X_pca = pca.transform(X_scaled)

        # Predictions
        predicted_salary = float(lr_model.predict(X_pca)[0])
        knn_pred = int(knn_model.predict(X_pca)[0])
        svm_pred = int(svm_model.predict(X_pca)[0])

        return jsonify({
            "predicted_salary_usd": round(predicted_salary, 2),
            "knn_classification": "Above Median" if knn_pred == 1 else "Below Median",
            "svm_classification": "Above Median" if svm_pred == 1 else "Below Median",
            "median_salary_usd": round(median_salary, 2),
            "model_scores": model_scores,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_models()
    app.run(debug=True, port=5000)
