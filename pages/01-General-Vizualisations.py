"""
General Visualisations:
    - Histogram (distribution + outliers, filter per column)
    - Pie chart (salary tiers)
    - Bar chart (job posts by year + growth metric)
    - Scatter plot (experience vs category/title)
    - Line chart (monthly demand trend with filters)
    - Skills frequency bar chart (+ top N + selected skills)
    - Bubble chart (demand score vs salary, size=growth, color=LLM binary)
    - Interpretation section
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_extras.let_it_rain import rain

st.set_page_config(page_title="General Visualisations", page_icon="📈", layout="wide")
st.title("📈 General Visualisations")

@st.cache_data
def load_data():
    file_path = "./EDA/ai_jobs_market_2025_2026.csv"
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Fișierul '{file_path}' nu a fost găsit.")
        return pd.DataFrame()

def pick_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["1", "true", "yes", "y", "llm", "t"])

df = load_data()
if df.empty:
    st.stop()

# --- Optional rain effect ---
rain_mode = st.radio("Rain effect", ["Off", "On"], index=0, horizontal=True)
if rain_mode == "On":
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



st.subheader("Histogram - distribution & outliers")
if num_cols_all:
    hist_col = st.selectbox("Numeric column", num_cols_all, index=0, key="hist_col_tab")
    bins = st.slider("Bins", 10, 100, 35, key="hist_bins_tab")
    fig_hist = px.histogram(filtered, x=hist_col, nbins=bins, title=f"Distribution of {hist_col}")
    st.plotly_chart(fig_hist, width="stretch")

    s = pd.to_numeric(filtered[hist_col], errors="coerce").dropna()
    if len(s) > 5:
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((s < low) | (s > high)).sum()
        st.caption(f"Outliers (IQR): {outliers} | Range: [{low:,.2f}, {high:,.2f}]")
else:
    st.info("Nu există coloane numerice.")

st.divider()

st.subheader("Pie Chart - salary tiers")
if salary_col:
    tmp = filtered[[salary_col]].copy()
    tmp[salary_col] = pd.to_numeric(tmp[salary_col], errors="coerce")
    tmp = tmp.dropna(subset=[salary_col])
    if not tmp.empty:
        tmp["salary_tier"] = pd.qcut(tmp[salary_col], q=4, labels=["Low", "Mid", "High", "Elite"], duplicates="drop")
        pie_df = tmp["salary_tier"].value_counts().reset_index()
        pie_df.columns = ["salary_tier", "count"]
        fig_pie = px.pie(pie_df, names="salary_tier", values="count", title="Salary tiers distribution")
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, width="stretch")
    else:
        st.info("Nu există salarii valide.")
else:
    st.info("Nu s-a găsit coloană de salariu.")

st.divider()

st.subheader("Bar Chart - number of job posts by year")
if year_col:
    year_counts = filtered[year_col].dropna().astype(int).value_counts().sort_index().reset_index()
    year_counts.columns = ["year", "posts"]
    fig_year = px.bar(year_counts, x="year", y="posts", text="posts", title="Job posts per year")
    st.plotly_chart(fig_year, width="stretch")

    if len(year_counts) >= 2 and year_counts.iloc[0]["posts"] > 0:
        first, last = year_counts.iloc[0], year_counts.iloc[-1]
        growth_pct = ((last["posts"] - first["posts"]) / first["posts"]) * 100
        st.metric("Growth from first to last year", f"{growth_pct:+.1f}%")
else:
    st.info("Nu există coloană de an.")

st.divider()

st.subheader("Scatterplot - experience vs category/title")
scatter_x_default = exp_col if exp_col in num_cols_all else (num_cols_all[0] if num_cols_all else None)
scatter_y_options = [c for c in [job_cat_col, job_title_col] if c] or obj_cols_all

if scatter_x_default and scatter_y_options:
    x_col = st.selectbox("X (numeric)", num_cols_all, index=num_cols_all.index(scatter_x_default), key="sc_x")
    y_col = st.selectbox("Y (category/title)", scatter_y_options, index=0, key="sc_y")

    sc_df = filtered[[x_col, y_col]].copy()
    sc_df[x_col] = pd.to_numeric(sc_df[x_col], errors="coerce")
    sc_df = sc_df.dropna(subset=[x_col, y_col])

    fig_sc = px.scatter(sc_df, x=x_col, y=y_col, color=y_col, opacity=0.7, title=f"{x_col} vs {y_col}")
    st.plotly_chart(fig_sc, width="stretch")
else:
    st.warning(f"Nu am suficiente coloane pentru scatter. Numerice: {num_cols_all}")

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

st.subheader("Interpretation")
c1, c2 = st.columns(2)
with c1:
    if salary_col and salary_col in filtered.columns:
        s = pd.to_numeric(filtered[salary_col], errors="coerce")
        st.markdown(f"- Average salary: **${s.mean():,.0f}**")
        st.markdown(f"- Median salary: **${s.median():,.0f}**")
    st.markdown(f"- Total filtered posts: **{len(filtered):,}**")
with c2:
    if demand_col and demand_col in filtered.columns:
        d = pd.to_numeric(filtered[demand_col], errors="coerce")
        st.markdown(f"- Mean demand score: **{d.mean():.2f}**")
    if job_title_col and job_title_col in filtered.columns:
        top_role = filtered[job_title_col].astype(str).value_counts().head(1)
        if len(top_role):
            st.markdown(f"- Most frequent role: **{top_role.index[0]}** ({int(top_role.iloc[0])} posts)")
    st.markdown("- Compare tabs using sidebar filters for consistent insights.")