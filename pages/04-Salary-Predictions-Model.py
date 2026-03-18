"""
Salary prediction page:
- Linear Regression
- Quadratic Regression
- Comparison on test set
"""

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Home import load_data

st.set_page_config(page_title="Salary Regression Models", layout="wide")


# -------------------------
# Helpers
# -------------------------
def clean_skills(text):
    if pd.isna(text):
        return []
    text = str(text).strip()
    if not text:
        return []
    return [s.strip().lower() for s in text.split("|") if s.strip()]


def prepare_dataset(df):
    df = df.copy()

    # target
    target_col = "annual_salary_usd"
    if target_col not in df.columns:
        raise ValueError("Column 'annual_salary_usd' not found in dataset.")

    # basic skill-derived feature
    if "required_skills" in df.columns:
        df["skills_list"] = df["required_skills"].apply(clean_skills)
        df["skill_count"] = df["skills_list"].apply(len)
    else:
        df["skill_count"] = 0

    # Drop obvious leakage / ID columns
    drop_cols = [
        "job_id",
        "annual_salary_usd",
        "salary_min_usd",
        "salary_max_usd",
        "salary_tier",
        "required_skills",
        "skills_list",
    ]

    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing_drop_cols)
    y = df[target_col]

    # remove rows with missing target
    valid_idx = y.notna()
    X = X.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()

    return X, y


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    # 1) Linear Regression
    linear_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    linear_model.fit(X_train, y_train)
    linear_preds = linear_model.predict(X_test)

    # 2) Quadratic Regression
    # PolynomialFeatures degree=2 after preprocessing
    quadratic_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("regressor", LinearRegression())
    ])

    quadratic_model.fit(X_train, y_train)
    quadratic_preds = quadratic_model.predict(X_test)

    linear_metrics = evaluate_regression(y_test, linear_preds)
    quadratic_metrics = evaluate_regression(y_test, quadratic_preds)

    return {
        "linear_model": linear_model,
        "quadratic_model": quadratic_model,
        "linear_metrics": linear_metrics,
        "quadratic_metrics": quadratic_metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "linear_preds": linear_preds,
        "quadratic_preds": quadratic_preds,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


def evaluate_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


@st.cache_resource
def load_and_train():
    df = load_data().copy()
    X, y = prepare_dataset(df)
    results = train_models(X, y)
    return df, X, y, results


# -------------------------
# Load data and train
# -------------------------
df, X, y, results = load_and_train()

linear_model = results["linear_model"]
quadratic_model = results["quadratic_model"]
linear_metrics = results["linear_metrics"]
quadratic_metrics = results["quadratic_metrics"]
numeric_features = results["numeric_features"]
categorical_features = results["categorical_features"]

# -------------------------
# Page
# -------------------------
st.title("Salary Prediction with Regression Models")

st.write(
    """
This page compares two regression models for predicting `annual_salary_usd`:
- Linear Regression
- Quadratic Regression

The target-leakage columns (`salary_min_usd`, `salary_max_usd`, `salary_tier`) are excluded.
"""
)

# -------------------------
# Metrics
# -------------------------
st.subheader("Model Performance on Test Set")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Linear RMSE", f"{linear_metrics['RMSE']:,.0f}")
with c2:
    st.metric("Linear MAE", f"{linear_metrics['MAE']:,.0f}")
with c3:
    st.metric("Linear R²", f"{linear_metrics['R2']:.3f}")

c4, c5, c6 = st.columns(3)
with c4:
    st.metric("Quadratic RMSE", f"{quadratic_metrics['RMSE']:,.0f}")
with c5:
    st.metric("Quadratic MAE", f"{quadratic_metrics['MAE']:,.0f}")
with c6:
    st.metric("Quadratic R²", f"{quadratic_metrics['R2']:.3f}")

# -------------------------
# Comparison table
# -------------------------
st.markdown("---")
st.subheader("Performance Comparison")

comparison_df = pd.DataFrame({
    "Model": ["Linear Regression", "Quadratic Regression"],
    "RMSE": [linear_metrics["RMSE"], quadratic_metrics["RMSE"]],
    "MAE": [linear_metrics["MAE"], quadratic_metrics["MAE"]],
    "R2": [linear_metrics["R2"], quadratic_metrics["R2"]],
})

st.dataframe(comparison_df, use_container_width=True)

# -------------------------
# Prediction form
# -------------------------
st.markdown("---")
st.subheader("Try a Salary Prediction")

with st.form("salary_prediction_form"):
    input_data = {}

    st.markdown("### Numeric Inputs")
    num_cols = st.columns(2)
    for i, col in enumerate(numeric_features):
        median_value = float(X[col].median()) if pd.api.types.is_numeric_dtype(X[col]) else 0.0
        input_data[col] = num_cols[i % 2].number_input(
            col,
            value=median_value
        )

    st.markdown("### Categorical Inputs")
    cat_cols = st.columns(2)
    for i, col in enumerate(categorical_features):
        options = sorted([v for v in X[col].dropna().astype(str).unique().tolist()])
        default_value = options[0] if options else ""
        input_data[col] = cat_cols[i % 2].selectbox(
            col,
            options=options,
            index=0 if options else None
        )

    submitted = st.form_submit_button("Predict Salary")

if submitted:
    input_df = pd.DataFrame([input_data])

    linear_salary = linear_model.predict(input_df)[0]
    quadratic_salary = quadratic_model.predict(input_df)[0]

    st.success("Prediction completed")

    p1, p2 = st.columns(2)
    with p1:
        st.metric("Linear Regression Prediction", f"${linear_salary:,.0f}")
    with p2:
        st.metric("Quadratic Regression Prediction", f"${quadratic_salary:,.0f}")

# -------------------------
# Preview
# -------------------------
st.markdown("---")
st.subheader("Model Input Preview")
preview_cols = [c for c in X.columns[:10]]
st.dataframe(pd.concat([X[preview_cols].head(10), y.head(10)], axis=1), use_container_width=True)