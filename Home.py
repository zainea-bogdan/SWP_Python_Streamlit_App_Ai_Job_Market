import streamlit as st
import pandas as pd
import plotly.express as px

# Aceasta sectiune configureaza titlul din tab-ul de la browser
st.set_page_config(page_title="AI Job Market 2025-2026 Overview", layout="wide")


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

# aici se citeste csv ul si este returneat ca un dataframe
df = load_data()


# Sectiunea de mai jos ajuta la setarea textului introductiv al pagini de home
st.title("Analiza Pietii Joburile cu AI (2025-2026)")

st.divider()

st.subheader("Proiect realizat de: Zainea Bogdan & Zorila Maria Cristina")
st.markdown("""
**Descriere** : Scopul acestei aplicații este de a demonstra capabilitățile oferite de Streamlit, utilizând setul de date ales ca suport pentru explorare și analiză.  
Deși datele sunt sintetice, inspirate totusi din surse reale de date, acestea permit simularea unor scenarii reale și evidențierea modului în care pot fi extrase insight-uri relevante în contextul recrutării din domeniul IT, in zona de joburilor dependente de AI.
""")

st.divider()

# Sectiunea de mai jos permite afisarea unui container cu 2 "sectiuni"/coloane.
with st.container():
    st.subheader("Sursa de Date & Credentials")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Dataset Owner:** Syed Ali Taqi")
        st.markdown("**Platform:** [Kaggle - AI Jobs Market 2025-2026 Salaries](https://www.kaggle.com/datasets/alitaqishah/ai-jobs-market-2025-2026-salaries)")
    with col_b:
        st.markdown("**Primary Data Sources:**")
        st.caption("Ravio Global Salary Benchmark, Robert Half Technology Guide, LinkedIn Workforce Insights, and Glassdoor.")

st.divider()

# Mai jos am incercat sa oferim pentru fiecare metrica cate o definitie ( tinuta in engleza for simplicity :) ) 
st.subheader("Descrierea coloanelor din dataset")

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

# punerea acestui dictionar intr un df
desc_df = pd.DataFrame(list(column_descriptions.items()), columns=["Column Name", "Description"])
st.table(desc_df)


st.divider()


# Mai jos este o linie cu 5 metrice care sa permita un "quick overview" asupra setului de date
if not df.empty:
    st.subheader("High-Level Market Metrics")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    m_col1.metric("Job Postings", len(df))
    m_col2.metric("AI/ML Roles", df['job_title'].nunique())
    m_col3.metric("Countries", df['country'].nunique())
    m_col4.metric("Industries", df['industry'].nunique())
    m_col5.metric("Company Sizes", df['company_size'].nunique())

st.divider()

# Mai jos este aratat un sample din datele originale
st.subheader("Dataset Preview")
if not df.empty:
    st.write("A sample of 10 random rows from the dataset:")
    st.dataframe(df.sample(10), use_container_width=True)

st.divider()

# Mai jos este aratat un sample din datele originale
st.subheader("Dataset Description:")
if not df.empty:
    st.write("Descrierea coloanelor numerice:")
    st.dataframe(df.describe(), use_container_width=True)

st.divider()



st.subheader("Analiză Univariată")
st.markdown("Explorează distribuția fiecărei variabile pentru a identifica tipare, valori extreme (outliers) sau anomalii.")

if not df.empty:
    # Organizăm selectorul într-o coloană mai îngustă
    col_sel, col_stats = st.columns([1, 2])
    
    with col_sel:
        filter_column = st.selectbox("Selectează coloana:", df.columns, index=0)
    
    series = df[filter_column].dropna()
    
    # --- Indicatori Rapizi (Quick Stats) ---
    with col_stats:
        m1, m2, m3 = st.columns(3)
        m1.metric("Valori Unice", series.nunique())
        m2.metric("Valori Lipsă (NaN)", df[filter_column].isna().sum())
        if pd.api.types.is_numeric_dtype(series):
            m3.metric("Medie", f"{series.mean():.2f}")
        else:
            m3.metric("Mod (Top)", series.mode()[0] if not series.empty else "N/A")

    st.divider()

    if pd.api.types.is_numeric_dtype(series):
        tab1, tab2 = st.tabs(["Histogramă", "Boxplot"])

        with tab1:
            bins = st.slider("Ajustează granulația (bins):", 5, 60, 20, key="hist_bins")
            
            # Adăugăm și o linie de tip KDE (densitate) pentru context
            fig_hist = px.histogram(
                df, x=filter_column, nbins=bins,
                title=f"Distribuția numerică: {filter_column}",
                color_discrete_sequence=['#3498DB'],
                opacity=0.7
            )
            fig_hist.update_layout(bargap=0.05, xaxis_title=filter_column, yaxis_title="Frecvență")
            st.plotly_chart(fig_hist, use_container_width=True)

        with tab2:
            fig_box = px.box(
                df, y=filter_column,
                title=f"Detectare Outliers: {filter_column}",
                points="all", # Vedem toate punctele pentru a observa grupările
                color_discrete_sequence=['#E74C3C']
            )
            st.plotly_chart(fig_box, use_container_width=True)

    else:
        # Pentru categorice, folosim o abordare de tip Pareto (cele mai frecvente)
        top_n = st.slider("Câte categorii de top afișăm?", 5, 30, 10)
        vc = series.value_counts().head(top_n).reset_index()
        vc.columns = [filter_column, "count"]

        fig_bar = px.bar(
            vc, x=filter_column, y="count",
            text="count", # Arată cifra direct pe bară
            title=f"Top {top_n} frecvențe: {filter_column}",
            color="count",
            color_continuous_scale='Blues'
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(xaxis_title=filter_column, yaxis_title="Număr apariții")
        st.plotly_chart(fig_bar, use_container_width=True)

# --- Secțiunea de Concluzii ---
st.subheader("Curios de ce am observat noi? :)")

with st.expander("Vezi detaliile analizei de tip 'Gold Rush' și 'Paradox'", expanded=True):
    st.markdown(f"""
    ### 1. Cele mai frecvente Joburi cu AI
    În urma analizei distribuției pentru **`job_title`**, am observat că top 3 cele mai "căutate" roluri sunt: **LLM Engineer**, **Robotics Engineer (AI)** și **Prompt Engineer**. 
    Acest lucru evidențiază faptul că obiceiurile noastre zilnice de a învăța să folosim mai eficient LLM-urile (ChatGPT, Gemini, Claude) ne pot influenta cariera pe viitor.

    ### 2. Nu tot AI este un LLM
    Dacă ne uităm la distribuția coloanei **`is_llm_role`**, observăm că aprox. **78%** din roluri **nu sunt** pur legate de LLM. 
    Asta sugerează că, deși este important să știm să folosim LLM-urile, paradigma actuală indică faptul că AI se îndreaptă într-o direcție mult mai autonoma și mai avansată față de ce suntem obișnuiți - search out for Ai Agent. 
    Daca nu ai auzit de AI Agents pana acum, ai trait sub o piatra or sth :).

    

    ### 3. Paradoxul Joburilor și al Experienței
    Observăm un aspect interesant la etichetele de salarii: foarte multe roluri sunt în tier-urile **Mid** până la **Elite**. 
    * **Paradoxul Etichetei:** Deși majoritatea sunt targetate ca *Entry-Level*, acest lucru se poate datora naturii semi-sintetice a datelor. 
    * **Realitatea Experienței:** Distribuția pe ani de experiență arată că **60%** din roluri necesită între **5-9 ani**, în timp ce categoria 0-4 ani reprezintă doar aproximativ **30%** (jumătate din categoria mid-tier).

    ### 4. Industriile Lider
    Principalele 3 industrii care caută roluri tehnice de AI sunt surprinzătoare: **Industria Automotivelor**, **Medicina** și **Guvernamental**, urmate de Finanțe și Energie.

    ### 5. Notă asupra Datelor Sintetice
    Trebuie să fim constant realiști: fiind un set de date sintetic, acesta servește mai mult ca o **sursă de inspirație și curiozitate** decât ca rezultate concrete. Deși interpretăm datele ca și când ar fi realiste, suntem conștienți de limitările cercetării și le folosim pentru a valida procesele de analiză.
    """)

# 7. Link to Next Page
st.divider()

st.markdown("### 🔍 Vrei sa descoperim mai mult?")
st.write("Apasa pe butonul de mai jos pentru a explora mai multe vizualizari, care pot dezvalui pattern-uri interesante ale setului de date de fata!.")

# Formulating the navigation button
if st.button("Let's Viz this data to get a better look at what's hiding"):
    try:
        st.switch_page("pages/01-General-Vizualisations.py")
    except Exception as e:
        st.error("Page not found. Please ensure the file exists in the 'pages/' folder.")