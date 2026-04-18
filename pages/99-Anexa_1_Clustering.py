"""
This page is intended to showcase: 
    - Clustering Model Roles by skills requirments.
    - Role Classification from required skills (NLP)
    - Salary Predictions using regression models (to be tested.)
    - To be disccused what can be done
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
st.set_page_config(page_title="Skill Clustering Demo", layout="centered")



# Functia de mai jos ajuta citirea csv ului cu date si face un try-catch ca sa vada daca path-ul dat este valid sau nu.
@st.cache_data
def load_data():
    file_path = "./EDA/ai_jobs_market_2025_2026.csv"
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it is in the working directory.")
        return pd.DataFrame()

# -------------------------
# Clean skills column
# -------------------------
def clean_skills(text):
    """
    Example input:
    'Python|SQL|Machine Learning|TensorFlow'

    Output:
    ['python', 'sql', 'machine learning', 'tensorflow']
    """
    if pd.isna(text):
        return []

    text = str(text).strip()

    if not text:
        return []

    skills = [s.strip().lower() for s in text.split("|") if s.strip()]
    return skills


# -------------------------
# Train clustering
# -------------------------
@st.cache_resource
def train_clustering():
    df = load_data()
    # imi verific daca am coloana de required skills
    if "required_skills" not in df.columns:
        raise ValueError("Column 'required_skills' not found in dataset.")

    # imi fac listele 
    df["skills_list"] = df["required_skills"].apply(clean_skills)

    # Keep only rows that actually have skills
    df = df[df["skills_list"].map(len) > 0].reset_index(drop=True)

    # mi am mapat fiecare valaore unica cu un label
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["skills_list"])

    # Train clustering model
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    # Build cluster profiles
    skill_df = pd.DataFrame(X, columns=mlb.classes_)
    skill_df["cluster"] = df["cluster"]

    cluster_profiles = skill_df.groupby("cluster").mean()

    cluster_top_skills = {}
    for cluster_id in cluster_profiles.index:
        top_skills = (
            cluster_profiles.loc[cluster_id]
            .sort_values(ascending=False)
            .head(10)
            .index
            .tolist()
        )
        cluster_top_skills[int(cluster_id)] = top_skills

    # Example job titles per cluster
    title_col = "job_title" if "job_title" in df.columns else None

    if title_col:
        cluster_examples = (
            df.groupby("cluster")[title_col]
            .apply(lambda x: list(x.value_counts().head(5).index))
            .to_dict()
        )
    else:
        cluster_examples = {}

    return df, mlb, kmeans, cluster_top_skills, cluster_examples


# -------------------------
# Predict cluster for new input
# -------------------------
def predict_cluster(user_input, mlb, kmeans):
    """
    User should also enter skills separated by |
    Example:
    python|sql|machine learning
    """
    user_skills = clean_skills(user_input)

    row = np.zeros(len(mlb.classes_), dtype=int)
    skill_to_idx = {skill: i for i, skill in enumerate(mlb.classes_)}

    matched = []
    unmatched = []

    for skill in user_skills:
        if skill in skill_to_idx:
            row[skill_to_idx[skill]] = 1
            matched.append(skill)
        else:
            unmatched.append(skill)

    if row.sum() == 0:
        return None, matched, unmatched

    cluster_id = int(kmeans.predict(row.reshape(1, -1))[0])
    return cluster_id, matched, unmatched


# -------------------------
# Run app
# -------------------------
df, mlb, kmeans, cluster_top_skills, cluster_examples = train_clustering()

st.title("Clustering Roles by Skill Requirements")
st.write("Enter skills separated by `|` to test which cluster they belong to.")

skills_input = st.text_area(
    "Skills",
    placeholder="python|sql|machine learning|tensorflow"
)

if st.button("Test Cluster"):
    cluster_id, matched, unmatched = predict_cluster(skills_input, mlb, kmeans)

    if cluster_id is None:
        st.error("None of the entered skills matched the dataset vocabulary.")
    else:
        st.success(f"This skill set belongs to Cluster {cluster_id}")
        st.write("**Matched skills:**", matched)
        st.write("**Unmatched skills:**", unmatched if unmatched else "None")
        st.write("**Top skills in this cluster:**", cluster_top_skills[cluster_id])

        if cluster_examples:
            st.write("**Example job titles in this cluster:**", cluster_examples.get(cluster_id, []))

st.markdown("---")
st.subheader("Explore clusters")

for cluster_id in sorted(cluster_top_skills.keys()):
    with st.expander(f"Cluster {cluster_id}"):
        st.write("**Top skills:**", cluster_top_skills[cluster_id])
        if cluster_examples:
            st.write("**Example job titles:**", cluster_examples.get(cluster_id, []))

st.markdown("---")
st.subheader("Preview parsed skills from dataset")
st.dataframe(df[["required_skills", "skills_list"]].head(10))