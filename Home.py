"""
This page is intended to:
    - presented the dataset taken into consideration
    - presented each column description 
    - giving credentials to owners and the datasource providers
    - general metrics customs: 
        - number of job posting (ex: 1500)
        - number of distinct AI/ML roles (ex: 25)
        - number of unique countries (ex: 14)
        - number of industries (ex: 12)
        - number of company sizes(ex: 5)
    - Showcasing a dataframe with sample data ( + loading into cache the dataframe).
    - Presenting the description of each column for this dataset. 
    - Analysis of the description output (what is unsual)*
    - Link to next page
"""
import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(page_title="AI Job Market 2025-2026 Overview", layout="wide")

# 1. Dataset Loading into Cache
@st.cache_data
def load_data():
    file_path = "./EDA/ai_jobs_market_2025_2026.csv"
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it is in the working directory.")
        return pd.DataFrame()

df = load_data()

# basically below is the title and a markdown paragraph which is like an introduction.
st.title("AI Jobs Market Analysis (2025-2026)")
st.subheader("Project made by: Zainea Bogdan & Zorila Maria Cristina")
st.markdown("""
This application explores a comprehensive dataset of Artificial Intelligence and Machine Learning job roles, 
focusing on projected salary trends, remote work availability, and the specific impact of LLM-related expertise 
on compensation across various industries.
""")

# defining a container, which is like a portion from the page where you can place the componenets as you wish in my case the first line is the title, then the second line is spli into 2 columns/cards with data into them
with st.container():
    st.subheader("Data Source & Credentials")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Dataset Owner:** Syed Ali Taqi")
        st.markdown("**Platform:** [Kaggle - AI Jobs Market 2025-2026 Salaries](https://www.kaggle.com/datasets/alitaqishah/ai-jobs-market-2025-2026-salaries)")
    with col_b:
        st.markdown("**Primary Data Sources:**")
        st.caption("Ravio Global Salary Benchmark, Robert Half Technology Guide, LinkedIn Workforce Insights, and Glassdoor.")

st.divider()

# below are the general metrics cards i mentioned in description
if not df.empty:
    st.subheader("High-Level Market Metrics")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    m_col1.metric("Job Postings", len(df))
    m_col2.metric("AI/ML Roles", df['job_title'].nunique())
    m_col3.metric("Countries", df['country'].nunique())
    m_col4.metric("Industries", df['industry'].nunique())
    m_col5.metric("Company Sizes", df['company_size'].nunique())

st.divider()

# 4. Showcasing Sample Data
st.subheader("Dataset Preview")
if not df.empty:
    st.write("A sample of 10 random rows from the dataset:")
    st.dataframe(df.sample(10), use_container_width=True)

st.divider()

# 5. Column Descriptions
st.subheader("Dataset Schema & Column Descriptions")

column_descriptions = {
    "job_id": "Unique identifier for each job entry.",
    "job_title": "The specific title of the AI/ML role (e.g., AI Agent Developer, LLM Engineer).",
    "job_category": "The broader category the job falls under (Engineering, Data Science, etc.).",
    "experience_level": "Seniority of the position (Junior, Mid, Senior, Lead).",
    "years_of_experience": "The number of years of professional experience required.",
    "education_required": "Minimum education qualification needed.",
    "annual_salary_usd": "The estimated annual base salary in USD.",
    "salary_min_usd / salary_max_usd": "The projected salary range for the role.",
    "city / country": "Geographic location of the hiring company.",
    "remote_work": "Categorization of work environment (On-site, Hybrid, Fully Remote).",
    "company_size": "Size of the organization (Startup, SME, Big Tech, etc.).",
    "industry": "The market sector of the company.",
    "required_skills": "Key technical and soft skills requested (separated by '|').",
    "ai_salary_premium_pct": "The percentage boost in salary attributed to AI-specific skills.",
    "demand_score": "A score (0-100) representing how competitive the role is.",
    "demand_growth_yoy_pct": "Projected year-over-year growth in demand for this role.",
    "is_llm_role": "Binary flag (1 = Yes, 0 = No) indicating if the role focuses on Large Language Models.",
    "salary_tier": "Binned classification of salary (e.g., Senior, Elite, Upper-Mid)."
}

# Convert dictionary to a nice table for display
desc_df = pd.DataFrame(list(column_descriptions.items()), columns=["Column Name", "Description"])
st.table(desc_df)

# 6. Analysis of Description Output (What is unusual)
st.subheader("Analysis: What is unusual?")
st.info("""
After reviewing the dataset schema and summary, several interesting points emerge:
1. **Future Projection (2025-2026):** Unlike historical datasets, this is focused on *future-looking* projections, including roles like 'AI Agent Developer' and 'Prompt Engineer' as core established positions.
2. **AI Salary Premium:** The inclusion of the `ai_salary_premium_pct` column is unique; it suggests that 'AI expertise' is treated as a measurable value-add on top of base engineering salaries.
3. **The LLM Binary:** The dataset specifically tracks `is_llm_role`, highlighting that Generative AI is no longer a sub-skill but a primary category for market segmentation.
4. **Salary Density:** The 'Elite' salary tier (>$300k) appears frequently, reflecting the high-stakes arms race for AI talent currently observed in the industry.
""")

# 7. Link to Next Page
st.divider()

st.markdown("### 🔍 Ready to dive deeper?")
st.write("Click the button below to explore interactive charts and hidden trends within the AI job market.")

# Formulating the navigation button
if st.button("Let's Viz this data to get a better look at what's hiding"):
    try:
        st.switch_page("pages/01-General-Vizualisations.py")
    except Exception as e:
        st.error("Page not found. Please ensure the file exists in the 'pages/' folder.")