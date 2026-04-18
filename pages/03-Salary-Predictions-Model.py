import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@st.cache_data
def load_data():
    file_path = "./EDA/ai_jobs_market_2025_2026.csv"
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Fișierul '{file_path}' nu a fost găsit.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

st.set_page_config(page_title="Salary Regression Models", layout="wide")

# -------------------------
# 1. Precise Data Preparation
# -------------------------
def prepare_dataset(df):
    df = df.copy()
    target_col = "annual_salary_usd"
    
    # 1. Curated Features: Focused on high-impact drivers
    selected_features = [
        "years_of_experience", "education_required", "job_category", "country", "city",
        "company_size", "posting_month",
        "remote_work", "industry", "job_title", "experience_level"
    ]
    
    # 2. Strict Leakage Prevention: Explicitly exclude salary categories/tiers
    leakage_cols = ["salary_tier", "salary_category", "salary_min_usd", "salary_max_usd"]
    
    existing_cols = [c for c in selected_features if c in df.columns]
    X = df[existing_cols].copy()
    
    # Ensure no leakage columns accidentally slipped into the feature set
    X = X.drop(columns=[c for c in leakage_cols if c in X.columns])
    
    y = df[target_col]

    # Clean rows with missing target
    valid_idx = y.notna()
    return X.loc[valid_idx].copy(), y.loc[valid_idx].copy()

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop='first', handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    
    return preprocessor, numeric_features, categorical_features

@st.cache_resource
def load_and_train():
    df = load_data().copy()
    X, y = prepare_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor, num_feats, cat_feats = build_preprocessor(X)

    # Model 1: Linear Regression
    lin_model = Pipeline([("prep", preprocessor), ("reg", LinearRegression())])
    lin_model.fit(X_train, y_train)
    
    # Model 2: Quadratic Regression
    quad_model = Pipeline([("prep", preprocessor), ("poly", PolynomialFeatures(2)), ("reg", LinearRegression())])
    quad_model.fit(X_train, y_train)

    return df, X, y, lin_model, quad_model, X_train, y_train, X_test, y_test, num_feats, cat_feats

# Execute Training
df, X, y, lin_model, quad_model, X_train, y_train, X_test, y_test, num_feats, cat_feats = load_and_train()

# -------------------------
# 2. Performance Comparison Metrics
# -------------------------
st.title("Salary Prediction Model Comparison")

lin_preds = lin_model.predict(X_test)
quad_preds = quad_model.predict(X_test)

metrics_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R2"],
    "Linear Regression": [
        np.sqrt(mean_squared_error(y_test, lin_preds)),
        mean_absolute_error(y_test, lin_preds),
        r2_score(y_test, lin_preds)
    ],
    "Quadratic Regression": [
        np.sqrt(mean_squared_error(y_test, quad_preds)),
        mean_absolute_error(y_test, quad_preds),
        r2_score(y_test, quad_preds)
    ]
}).set_index("Metric")

# HIGHLIGHTING LOGIC: Green for better performance
def highlight_best(row):
    if row.name in ["RMSE", "MAE"]:
        best_val = row.min()
    else: # For R2
        best_val = row.max()
    return ['background-color: #1e4620; color: white' if v == best_val else '' for v in row]

styled_df = metrics_df.style.apply(highlight_best, axis=1).format("{:,.2f}")

st.subheader("📊 Model Performance Comparison")
c1, c2, c3 = st.columns(3)
with c1: st.metric("Top R² Score", f"{metrics_df.loc['R2'].max():.3f}")
with c2: st.metric("Best RMSE", f"${metrics_df.loc['RMSE'].min():,.0f}")
with c3: st.metric("Best MAE", f"${metrics_df.loc['MAE'].min():,.0f}")

st.dataframe(styled_df, use_container_width=True)

st.markdown("---")
st.subheader("📝 Metric Definitions & Interpretation")

with st.expander("What do these metrics mean for my salary prediction?"):
    st.markdown(f"""
    * **RMSE (${metrics_df.loc['RMSE'].min():,.2f}):** The standard deviation of prediction errors. It penalizes large outliers.
    * **MAE (${metrics_df.loc['MAE'].min():,.2f}):** The average 'dollar amount' error per prediction.
    * **$R^2$ ({metrics_df.loc['R2'].max():.1%}):** The percentage of salary variation explained by the model features.
    """)
    st.info("The green highlight indicates the winning model for each metric.")

# -------------------------
# 3. Interactive Prediction
# -------------------------
st.markdown("---")
st.subheader("🎯 Final Salary Prediction")

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# Geography Logic
countries = sorted(X['country'].unique())
selected_country = row1_col1.selectbox("Select Country", options=countries)
available_cities = sorted(X[X['country'] == selected_country]['city'].unique())
selected_city = row1_col2.selectbox("Select City", options=available_cities)

# Job Category Logic
category_options = ["All Categories"] + sorted(X['job_category'].unique().tolist())
selected_category = row2_col1.selectbox("Select Job Category", options=category_options)

if selected_category == "All Categories":
    available_titles = sorted(X['job_title'].unique())
else:
    available_titles = sorted(X[X['job_category'] == selected_category]['job_title'].unique())
selected_title = row2_col2.selectbox("Select Job Title", options=available_titles)

with st.form("prediction_form"):
    actual_category = selected_category
    if selected_category == "All Categories":
        actual_category = X[X['job_title'] == selected_title]['job_category'].iloc[0]

    input_data = {
        'country': selected_country, 'city': selected_city,
        'job_category': actual_category, 'job_title': selected_title
    }

    c1, c2 = st.columns(2)
    input_data['years_of_experience'] = c1.number_input("Years of Experience", value=6.0)
    input_data['posting_month'] = c2.slider("Posting Month", 1, 12, 3)
    input_data['experience_level'] = c1.selectbox("Experience Level", options=sorted(X['experience_level'].unique()))
    input_data['education_required'] = c2.selectbox("Education Required", options=sorted(X['education_required'].unique()))
    input_data['remote_work'] = c2.selectbox("Remote Work Type", options=sorted(X['remote_work'].unique()))
    input_data['industry'] = c1.selectbox("Industry", options=sorted(X['industry'].unique()))
    input_data['company_size'] = c2.selectbox("Company Size", options=sorted(X['company_size'].unique()))

    predict_submit = st.form_submit_button("Predict Salary")

if predict_submit:
    input_df = pd.DataFrame([input_data])[X.columns]
    final_lin = lin_model.predict(input_df)[0]
    final_quad = quad_model.predict(input_df)[0]
    
    st.success(f"Linear Prediction: **${final_lin:,.0f}**")
    st.info(f"Quadratic Prediction: **${final_quad:,.0f}**")

