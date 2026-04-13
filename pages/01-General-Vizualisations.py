import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_extras.let_it_rain import rain

st.set_page_config(page_title="General Visualisations", layout="wide")
st.title(" General Visualisations")


# Am ales sa facem si aici functia de incarcare a datelor intrucat sa nu existe riscul ca daca o importam din home sa ne incarce cumva si pagina de home cu totul (testat ca se intamplat asta :) ) 
@st.cache_data
def load_data():
    file_path = "./EDA/ai_jobs_market_2025_2026.csv"
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Fișierul '{file_path}' nu a fost găsit.")
        return pd.DataFrame()

# aceasta functie contributie la selectia coloanelor care exista in dataframe si pe care le dam noi ca fiind cele dorite.
def pick_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# aceasta functie imi intoarce o serie de true sau false pentru acele coloane care sunt variabile categorialesi care trebuie converite in true sau false
def to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["1", "true", "yes", "y", "llm", "t"])

#========================================

# Aici este intrare in pagina de general Vizs in care exploram mai multe vizualirile si asa mai departe. Prima data as usual incarcam datele si daca cumva nu avem nimic in dataframe oprim rularea pagini
df = load_data()
if df.empty:
    st.stop()
st.divider()
# Odata cu venirea primavaeri am ales sa introducem si un mic efect cute in cazul in care vreti sa simti the cherry blossom season in our app :)
st.subheader("Do you like Cherry Blossoms?")
rain_mode = st.radio("Choose carefully: ", ["No", "Yes"], index=0, horizontal=True)
if rain_mode == "Yes":
    rain(emoji="🌸", font_size=42, falling_speed=5, animation_length="infinite")

# Resolve columns (more robust)
salary_col = pick_col(df, ["annual_salary_usd", "salary_usd", "salary", "avg_salary_usd"])
year_col = pick_col(df, ["posting_year", "year"])
month_col = pick_col(df, ["posting_month", "month"])
exp_col = pick_col(df, ["years_experience", "experience_years", "experience", "years_exp"])
job_cat_col = pick_col(df, ["job_category", "category", "role_category"])
job_title_col = pick_col(df, ["job_title", "title", "role_title"])
demand_col = pick_col(df, ["demand_score", "demand", "market_demand"])
growth_col = pick_col(df, ["demand_growth", "growth", "growth_rate", "trend_growth"])
skills_col = pick_col(df, ["required_skills", "skills", "key_skills", "skill_list"])
llm_col = pick_col(df, ["llm_related", "is_llm_role", "llm_flag", "llm", "requires_llm"])

# Prepare year/month from date if needed
date_col = pick_col(df, ["posting_date", "date", "created_at"])
if (year_col is None or month_col is None) and date_col is not None:
    dt = pd.to_datetime(df[date_col], errors="coerce")
    if year_col is None:
        df["posting_year"] = dt.dt.year
        year_col = "posting_year"
    if month_col is None:
        df["posting_month"] = dt.dt.month
        month_col = "posting_month"

# Global filter
filtered = df.copy()
if year_col:
    years = sorted(filtered[year_col].dropna().astype(int).unique().tolist())
    if years:
        selected_years = st.sidebar.multiselect("Filter year", years, default=years)
        filtered = filtered[filtered[year_col].isin(selected_years)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Rows after filters: {len(filtered):,}")

# Helper lists
num_cols_all = filtered.select_dtypes(include=[np.number]).columns.tolist()
obj_cols_all = filtered.select_dtypes(exclude=[np.number]).columns.tolist()


st.divider()

st.subheader("Distribuția Categoriilor de Salarii - pe Quartile")

if salary_col:
    tmp = filtered[[salary_col]].copy()
    tmp[salary_col] = pd.to_numeric(tmp[salary_col], errors="coerce")
    tmp = tmp.dropna(subset=[salary_col])
    
    if not tmp.empty:
        # 1. Calcularea Tier-urilor
        tmp["salary_tier"] = pd.qcut(tmp[salary_col], q=4, labels=["Low", "Mid", "High", "Elite"], duplicates="drop")
        pie_df = tmp["salary_tier"].value_counts().reset_index()
        pie_df.columns = ["salary_tier", "count"]
        
        # 2. Definirea culorilor pentru un aspect profesional
        color_map = {
            "Low": "#E74C3C",    # Roșu
            "Mid": "#F39C12",    # Portocaliu
            "High": "#3498DB",   # Albastru
            "Elite": "#27AE60"   # Verde
        }

        # 3. Crearea graficului tip Donut (hole=0.4)
        fig_pie = px.pie(
            pie_df, 
            names="salary_tier", 
            values="count", 
            title="Distribuția pe Nivele Salariale",
            color="salary_tier",
            color_discrete_map=color_map,
            hole=0.4,  # Transformă Pie în Donut
            height=600
        )
        
        fig_pie.update_traces(
            textposition="inside", 
            textinfo="percent+label",
            marker=dict(line=dict(color='#000000', width=2)) # Border pentru contrast
        )
        
        fig_pie.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            margin=dict(t=50, b=20, l=20, r=20)
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        # 4. Insights îmbunătățite
        st.info(f"""
        ### 💡 Insights:
        * **Confirmare Date Sintetice:** Observăm o distribuție perfect egală de **25%** pentru fiecare categorie. Acest lucru apare deoarece am folosit `pd.qcut`, care împarte datele în segmente egale bazate pe numărul de înregistrări (quartile).
        * **Praguri Salariale:** Într-un set de date reale, am vedea o "cocoașă" în zona de Mid/High și mult mai puține înregistrări la Elite. Aici, echilibrul perfect sugerează o generare algoritmică menită să acopere toate intervalele uniform.
        * **Utilitate:** Această împărțire ne ajută să vedem care sunt joburile care reușesc să "sară" în top 25% (Elite) indiferent de volumul total de date.
        """)
    else:
        st.info("Nu există salarii valide.")
else:
    st.info("Nu s-a găsit coloană de salariu.")

st.divider()

st.subheader("Bar Chart - Numarul de joburi cu AI, in 2026 fata de 2025")
if year_col:
    year_counts = filtered[year_col].dropna().astype(int).value_counts().sort_index().reset_index()
    year_counts.columns = ["year", "posts"]
    fig_year = px.bar(year_counts, x="year", y="posts", text="posts", title="Job posts per year")
    st.plotly_chart(fig_year, use_container_width=True) # Changed from width="stretch" to the standard parameter

    if len(year_counts) >= 2 and year_counts.iloc[0]["posts"] > 0:
        first, last = year_counts.iloc[0], year_counts.iloc[-1]
        growth_pct = ((last["posts"] - first["posts"]) / first["posts"]) * 100
        st.metric(
            label="Growth from first to last year", 
            value=f"{last['posts']} Posts", 
            delta=f"{growth_pct:+.1f}%"
        )
else:
    st.info("Nu există coloană de an.")

st.divider()

st.subheader("Scatterplot - Raportarea Categoriei de Joburi in relatie cu salariul anual")

# 1. Definirea coloanelor favorite
fav_x = "annual_salary_usd"
fav_y = "job_category"

# Opțiunile pentru Y (deja calculate de tine)
scatter_y_options = [c for c in [job_cat_col, job_title_col] if c] or obj_cols_all

if num_cols_all and scatter_y_options:
    # 2. Calcularea indexului pentru X (annual_salary_usd)
    try:
        default_ix_x = num_cols_all.index(fav_x)
    except ValueError:
        default_ix_x = 0

    # 3. Calcularea indexului pentru Y (job_category)
    try:
        default_ix_y = scatter_y_options.index(fav_y)
    except ValueError:
        default_ix_y = 0

    # 4. Crearea selectbox-urilor cu valorile default setate
    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("Selectează axa X (Numeric)", num_cols_all, index=default_ix_x, key="sc_x")
    with c2:
        y_col = st.selectbox("Selectează axa Y (Categorie)", scatter_y_options, index=default_ix_y, key="sc_y")

    # 5. Pregătirea datelor
    sc_df = filtered[[x_col, y_col]].copy()
    sc_df[x_col] = pd.to_numeric(sc_df[x_col], errors="coerce")
    sc_df = sc_df.dropna(subset=[x_col, y_col])
    
    # Sortăm după salariu pentru a ajuta vizualizarea dacă e cazul
    sc_df = sc_df.sort_values(by=x_col)

    # 6. Generarea graficului
    fig_sc = px.scatter(
        sc_df, 
        x=x_col, 
        y=y_col, 
        color=y_col, 
        opacity=0.7, 
        height=600, # Am mărit înălțimea pentru lizibilitate
        title=f"Analiză: {x_col} vs {y_col}",
        labels={x_col: "Salariu Anual (USD)", y_col: "Categorie Job"}
    )
    
    # Îmbunătățiri vizuale
    fig_sc.update_layout(showlegend=False) # Legendă ascunsă dacă culorile sunt pe Y
    fig_sc.update_yaxes(categoryorder="total ascending") # Ordonare după volumul de date

    st.plotly_chart(fig_sc, use_container_width=True)

    # --- Secțiunea de Insights ---
    st.info("""
    ### 💡 Insights:
    1. **Discrepante in Guvernanta Datelor si "AI Gold Rush":**
        Putem observa ca desi cam toata lumea vrea roluri de "AI Engineer", nu se acorda suficienta atentie **GUVERNANTEI** datelor. Fara date curate si mascate adecvat, AI-ul nu poate oferi raspunsuri de calitate.
    
    2. **Discrepante intre "Shiny Objects" si roluri esentiale:** Exista o balanta fragila; AI Engineers ating praguri de 380k USD, in timp ce Data Engineers (fara de care AI-ul nu ar avea date) plafoneaza mult mai jos (270k-280k). Rolurile de Data Governance sunt si mai subevaluate.
    
    3. **LLM e baza:** Predominanta rolurilor de LLM Engineer fata de AI Solution Architects sau Data Engineers este un semn clar al trendului actual de piata.
    
    4. **Note:** Aceste date sunt un semnal de alarma asupra problemelor de structura din departamentele de date care ar putea fi mascate de hype-ul actual.
    """)

else:
    st.warning("Nu am suficiente coloane pentru a genera graficul.")

st.divider()

st.subheader("Line chart - monthly demand trend")
if month_col and demand_col:
    line_df = filtered.copy()
    line_df[month_col] = pd.to_numeric(line_df[month_col], errors="coerce")
    line_df[demand_col] = pd.to_numeric(line_df[demand_col], errors="coerce")
    line_df = line_df.dropna(subset=[month_col, demand_col])

    filter_col = None
    if job_title_col:
        filter_col = job_title_col
    elif job_cat_col:
        filter_col = job_cat_col

    if filter_col:
        opts = sorted(line_df[filter_col].dropna().astype(str).unique().tolist())
        selected = st.multiselect(f"Filter {filter_col}", opts, default=opts[:5] if opts else [])
        if selected:
            line_df = line_df[line_df[filter_col].astype(str).isin(selected)]

    trend = line_df.groupby(month_col, as_index=False)[demand_col].mean().sort_values(month_col)
    fig_line = px.line(trend, x=month_col, y=demand_col, markers=True, title="Average demand score by month")
    st.plotly_chart(fig_line, width="stretch")
else:
    st.info("Lipsesc coloane pentru month + demand.")

st.divider()

st.subheader("Skills frequency")
if skills_col:
    sk = filtered[skills_col].dropna().astype(str)
    all_skills = []
    for txt in sk:
        all_skills.extend([p.strip().lower() for p in re.split(r"[,;|/]", txt) if p.strip()])

    if all_skills:
        skills_freq = pd.Series(all_skills).value_counts().reset_index()
        skills_freq.columns = ["skill", "count"]

        top_n = st.slider("Top N skills", 5, 50, 15, key="skills_top_tab")
        skill_options = skills_freq["skill"].head(100).tolist()
        selected_skills = st.multiselect("Inspect skills", skill_options, default=[])

        plot_skills = skills_freq[skills_freq["skill"].isin(selected_skills)] if selected_skills else skills_freq.head(top_n)
        fig_sk = px.bar(plot_skills, x="count", y="skill", orientation="h", title="Skills frequency")
        fig_sk.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_sk, width="stretch")
    else:
        st.info("Nu s-au extras skill-uri.")
else:
    st.info("Nu există coloană de skill-uri.")

st.divider()

st.subheader("Bubble chart - demand score x salary")
if len(num_cols_all) >= 3:
    # defaults if detected, otherwise fallback to first numeric cols
    x_default = demand_col if demand_col in num_cols_all else num_cols_all[0]
    y_default = salary_col if salary_col in num_cols_all else num_cols_all[min(1, len(num_cols_all)-1)]
    s_default = growth_col if growth_col in num_cols_all else num_cols_all[min(2, len(num_cols_all)-1)]

    x_col = st.selectbox("X", num_cols_all, index=num_cols_all.index(x_default), key="bub_x")
    y_col = st.selectbox("Y", num_cols_all, index=num_cols_all.index(y_default), key="bub_y")
    size_col = st.selectbox("Bubble size", num_cols_all, index=num_cols_all.index(s_default), key="bub_s")

    bub = filtered.copy()
    for c in [x_col, y_col, size_col]:
        bub[c] = pd.to_numeric(bub[c], errors="coerce")
    bub = bub.dropna(subset=[x_col, y_col, size_col])

    # keep bubble sizes positive
    bub["_bubble_size"] = bub[size_col].abs() + 1e-6

    if llm_col and llm_col in bub.columns:
        bub["color_group"] = to_bool_series(bub[llm_col]).map({True: "LLM", False: "Non-LLM"})
    elif job_cat_col and job_cat_col in bub.columns:
        bub["color_group"] = bub[job_cat_col].astype(str)
    else:
        bub["color_group"] = "All"

    hover_cols = [c for c in [job_title_col, job_cat_col] if c and c in bub.columns]
    fig_bub = px.scatter(
        bub,
        x=x_col,
        y=y_col,
        size="_bubble_size",
        color="color_group",
        hover_data=hover_cols,
        title=f"{x_col} vs {y_col} (size={size_col})",
    )
    st.plotly_chart(fig_bub, width="stretch")
else:
    st.warning(f"Bubble chart necesită minim 3 coloane numerice. Găsite: {num_cols_all}")

st.divider()

