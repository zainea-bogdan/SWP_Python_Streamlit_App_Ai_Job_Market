import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time

st.set_page_config(page_title="Advanced Visualisations", layout="wide")
st.title("Advanced Visualisations")

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

# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("Filtre")
industry_sel = st.sidebar.multiselect(
    "Industry",
    options=sorted(df["industry"].dropna().unique()),
    default=sorted(df["industry"].dropna().unique())
)
exp_sel = st.sidebar.multiselect(
    "Experience Level",
    options=sorted(df["experience_level"].dropna().unique()),
    default=sorted(df["experience_level"].dropna().unique())
)
remote_sel = st.sidebar.multiselect(
    "Remote Work",
    options=sorted(df["remote_work"].dropna().unique()),
    default=sorted(df["remote_work"].dropna().unique())
)
salary_range = st.sidebar.slider(
    "Annual Salary USD",
    int(df["annual_salary_usd"].min()),
    int(df["annual_salary_usd"].max()),
    (int(df["annual_salary_usd"].min()), int(df["annual_salary_usd"].max()))
)

filtered = df[
    (df["industry"].isin(industry_sel)) &
    (df["experience_level"].isin(exp_sel)) &
    (df["remote_work"].isin(remote_sel)) &
    (df["annual_salary_usd"].between(salary_range[0], salary_range[1]))
].copy()

st.subheader("Preview date filtrate")
st.dataframe(filtered.sample(min(10, len(filtered))), width="stretch")

if filtered.empty:
    st.warning("Nu există date după filtrele selectate.")
    st.stop()

# ----------------------------
# 1) Choropleth - roles per country
# ----------------------------
st.divider()
st.subheader("🌍 Geo Discrepancies - Număr roluri per țară")

country_counts = (
    filtered.groupby("country", dropna=False)
    .size()
    .reset_index(name="roles_count")
    .dropna(subset=["country"])
)

# Mapare country -> ISO-3 (fără dependențe noi)
_gap = px.data.gapminder()[["country", "iso_alpha"]].drop_duplicates()
_country_to_iso3 = dict(zip(_gap["country"], _gap["iso_alpha"]))
_alias_iso3 = {
    # Asia & Middle East (Locațiile tale cheie)
    "Singapore": "SGP",
    "UAE": "ARE",
    "United Arab Emirates": "ARE",
    "Japan": "JPN",
    "China": "CHN",
    "India": "IND",
    
    # Europe
    "UK": "GBR",
    "United Kingdom": "GBR",
    "Germany": "DEU",
    "France": "FRA",
    "Netherlands": "NLD",
    "Switzerland": "CHE",
    
    # North America & Oceania
    "USA": "USA",
    "United States": "USA",
    "Canada": "CAN",
    "Australia": "AUS",
    
    # Fallbacks pentru siguranță
    "Russia": "RUS",
    "South Korea": "KOR"
}

# Aplicare mapare
country_counts["iso3"] = country_counts["country"].map(_country_to_iso3)
country_counts["iso3"] = country_counts["iso3"].fillna(country_counts["country"].map(_alias_iso3))

# --- IMPORTANT: Debugging pentru a vedea ce nu apare ---
missing_countries = country_counts[country_counts["iso3"].isna()]["country"].unique()
if len(missing_countries) > 0:
    st.warning(f"⚠️ Următoarele țări nu sunt mapate corect și nu vor apărea pe hartă: {missing_countries}")


country_counts["iso3"] = country_counts["country"].map(_country_to_iso3)
country_counts["iso3"] = country_counts["iso3"].fillna(country_counts["country"].map(_alias_iso3))
country_counts = country_counts.dropna(subset=["iso3"])

fig_map = px.choropleth(
    country_counts,
    locations="iso3",
    locationmode="ISO-3",
    color="roles_count",
    hover_name="country",
    color_continuous_scale="Blues", # Păstrăm scara de albastru
    # --- ADAUGĂ ACEASTĂ LINIE pentru a face scara logaritmică ---
    color_continuous_midpoint=country_counts["roles_count"].median(), # Ajută la contrast
    title="Număr de joburi AI/ML per țară"
)
fig_map.update_layout(margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig_map, width="stretch")

# --- Globe interactiv pe baza lat/lon ---
st.markdown("### 🌐 Glob interactiv - selecție orașe")

@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def geocode_cities_nominatim(city_country_pairs) -> pd.DataFrame:
    out = []
    headers = {"User-Agent": "SWP-Streamlit/1.0 (student project)"}

    for city, country in city_country_pairs:
        city = str(city).strip()
        country = str(country).strip()

        # 1) query city + country
        queries = [f"{city}, {country}" if country else city]
        # 2) fallback query city only
        if country:
            queries.append(city)

        found = False
        for q in queries:
            try:
                r = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": q, "format": "json", "limit": 1},
                    headers=headers,
                    timeout=15,
                )
                if r.status_code == 200:
                    data = r.json()
                    if data:
                        out.append(
                            {
                                "city": city,
                                "country": country,
                                "latitude": float(data[0]["lat"]),
                                "longitude": float(data[0]["lon"]),
                            }
                        )
                        found = True
                        break
            except Exception:
                pass

        time.sleep(1.1)  # politeness for Nominatim
    return pd.DataFrame(out)


lat_candidates = ["latitude", "lat", "city_latitude", "job_latitude"]
lon_candidates = ["longitude", "lon", "lng", "city_longitude", "job_longitude"]

# folosim un df separat pentru partea geo (poate fi îmbogățit cu Google)
geo_source_df = filtered.copy()
lat_col = next((c for c in lat_candidates if c in geo_source_df.columns), None)
lon_col = next((c for c in lon_candidates if c in geo_source_df.columns), None)

# dacă nu există lat/lon, încearcă geocodare după city+country (Nominatim only)
if not (lat_col and lon_col) and {"city", "country"}.issubset(geo_source_df.columns):
    req_df = geo_source_df[["city", "country"]].dropna(subset=["city"]).copy()
    req_df["city"] = req_df["city"].astype(str).str.strip()
    req_df["country"] = req_df["country"].fillna("").astype(str).str.strip()

    # curățare orașe ne-geocodabile
    bad_city_values = {"", "remote", "hybrid", "unknown", "n/a", "na", "worldwide", "global"}
    req_df = req_df[~req_df["city"].str.lower().isin(bad_city_values)]

    # geocode doar top orașe după frecvență (mai rapid și util)
    top_req = (
        req_df.value_counts(["city", "country"])
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .head(60)
    )
    pairs = tuple(map(tuple, top_req[["city", "country"]].values.tolist()))

    with st.spinner("Geocoding orașe (Nominatim)..."):
        lookup = geocode_cities_nominatim(pairs)

    if not lookup.empty:
        geo_source_df["city"] = geo_source_df["city"].astype(str).str.strip()
        geo_source_df["country"] = geo_source_df["country"].fillna("").astype(str).str.strip()
        geo_source_df = geo_source_df.merge(lookup, on=["city", "country"], how="left")
        lat_col, lon_col = "latitude", "longitude"
        st.caption(f"Coordonate obținute prin Nominatim: {lookup.shape[0]} orașe")
    else:
        st.warning("Nominatim nu a returnat coordonate pentru orașele filtrate.")

if lat_col and lon_col and "city" in geo_source_df.columns:
    geo_df = geo_source_df[["city", "country", lat_col, lon_col]].copy()
    geo_df[lat_col] = pd.to_numeric(geo_df[lat_col], errors="coerce")
    geo_df[lon_col] = pd.to_numeric(geo_df[lon_col], errors="coerce")
    geo_df = geo_df.dropna(subset=["city", lat_col, lon_col])

    city_geo = (
        geo_df.groupby(["city", "country", lat_col, lon_col], dropna=False)
        .size()
        .reset_index(name="roles_count")
        .sort_values("roles_count", ascending=False)
    )

    if city_geo.empty:
        st.info("Nu există suficiente date geo valide (lat/lon) pentru orașe.")
    else:
        max_cities = min(100, len(city_geo))
        top_n_globe = st.slider("Număr orașe pe glob", 10, max_cities, min(40, max_cities))
        city_geo_top = city_geo.head(top_n_globe)

        selected_cities = st.multiselect(
            "Selectează orașele de afișat pe glob:",
            options=city_geo_top["city"].tolist(),
            default=city_geo_top["city"].head(1).tolist()
        )

        city_geo_plot = city_geo_top[city_geo_top["city"].isin(selected_cities)]

        fig_globe = px.scatter_geo(
            city_geo_plot,
            lat=lat_col,
            lon=lon_col,
            size="roles_count",
            color="roles_count",
            hover_name="city",
            custom_data=["city", "country"],  # pentru selecție
            hover_data={"country": True, "roles_count": True, lat_col: False, lon_col: False},
            color_continuous_scale="Plasma",
            projection="orthographic",
            title="Glob interactiv: distribuția rolurilor pe orașe"
        )
        fig_globe.update_geos(
            showland=True,
            landcolor="rgb(230, 230, 230)",
            showocean=True,
            oceancolor="rgb(180, 210, 255)",
            showcountries=True
        )
        fig_globe.update_layout(height=700, margin=dict(l=10, r=10, t=60, b=10))

        event = st.plotly_chart(
            fig_globe,
            width="stretch",
            key="globe_chart",
            on_select="rerun",
            selection_mode="points",
        )

        # --- Oraș selectat (click pe glob) ---
        selected_city = None
        selected_country = None

        try:
            points = event.get("selection", {}).get("points", []) if event else []
            if points:
                cd = points[0].get("customdata", [])
                if len(cd) >= 2:
                    selected_city = str(cd[0]).strip()
                    selected_country = str(cd[1]).strip()
        except Exception:
            pass

        # fallback manual dacă nu e click
        if not selected_city:
            city_options = (
                city_geo_top[["city", "country"]]
                .drop_duplicates()
                .apply(lambda r: f"{r['city']} | {r['country']}", axis=1)
                .tolist()
            )
            picked = st.selectbox("Selectează oraș pentru metrici:", ["(niciunul)"] + city_options)
            if picked != "(niciunul)":
                selected_city, selected_country = [x.strip() for x in picked.split("|", 1)]

        if selected_city:
            base = filtered.copy()
            base["city"] = base["city"].astype(str).str.strip()
            base["country"] = base["country"].fillna("").astype(str).str.strip()

            city_df = base[(base["city"] == selected_city) & (base["country"] == selected_country)].copy()

            if not city_df.empty:
                st.markdown(f"#### 📍 Metrici pentru {selected_city}, {selected_country}")

                roles_n = len(city_df)
                avg_salary = city_df["annual_salary_usd"].mean()
                med_salary = city_df["annual_salary_usd"].median()
                uniq_titles = city_df["job_title"].nunique()

                remote_share = (
                    city_df["remote_work"].astype(str).str.lower().isin(["true", "1", "yes", "remote"]).mean() * 100
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Nr. roluri", f"{roles_n:,}")
                c2.metric("Salariu mediu (USD)", f"{avg_salary:,.0f}")
                c3.metric("Salariu median (USD)", f"{med_salary:,.0f}")
                c4.metric("Job titles unice", f"{uniq_titles}")

                top_jobs = city_df["job_title"].value_counts().head(5).reset_index()
                top_jobs.columns = ["job_title", "count"]

                top_ind = city_df["industry"].value_counts().head(5).reset_index()
                top_ind.columns = ["industry", "count"]

                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption("Top job titles")
                    st.dataframe(top_jobs, width="stretch")
                with col_b:
                    st.caption("Top industrii")
                    st.dataframe(top_ind, width="stretch")
else:
    st.info("Nu există coloane lat/lon și nici geocodare Google disponibilă.")


# ----------------------------
# 2) Sunburst Industry -> selectable level
# ----------------------------
st.divider()
st.subheader("☀️ Sunburst - Repartiție în industrie")

level_2 = st.radio(
    "Alege nivelul 2:",
    ["job_category", "job_title"],
    horizontal=True
)

sunburst_data = (
    filtered.groupby(["industry", level_2], dropna=False)
    .size()
    .reset_index(name="count")
    .dropna(subset=["industry", level_2])
)

fig_sunburst = px.sunburst(
    sunburst_data,
    path=["industry", level_2],
    values="count",
    title=f"Industry → {level_2}"
)
fig_sunburst.update_layout(
    height=800,   # mareste inaltimea
    margin=dict(l=10, r=10, t=60, b=10)
)
st.plotly_chart(fig_sunburst, width="stretch")

# ----------------------------
# 3) Sunburst skills per job title
# ----------------------------
st.divider()
st.subheader("🧠 Sunburst - Skill-uri unice per job title")

skills_df = filtered[["job_title", "required_skills"]].dropna().copy()
skills_df["required_skills"] = skills_df["required_skills"].astype(str)

skills_df["skill"] = skills_df["required_skills"].str.split("|")
skills_df = skills_df.explode("skill")
skills_df["skill"] = skills_df["skill"].str.strip()
skills_df = skills_df[skills_df["skill"] != ""]

mode = st.radio(
    "Agregare skill-uri:",
    ["Apariții totale", "Skill-uri unice per job title"],
    horizontal=True
)

if mode == "Apariții totale":
    agg_skills = (
        skills_df.groupby(["job_title", "skill"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
else:
    # fiecare (job_title, skill) apare o singură dată
    agg_skills = (
        skills_df.drop_duplicates(subset=["job_title", "skill"])
        .groupby(["job_title", "skill"])
        .size()
        .reset_index(name="count")
    )

top_titles = st.slider("Număr job titles incluse", 5, 25, 12)
top_job_titles = (
    filtered["job_title"].value_counts().head(top_titles).index.tolist()
)
agg_skills = agg_skills[agg_skills["job_title"].isin(top_job_titles)]

fig_skills = px.sunburst(
    agg_skills,
    path=["job_title", "skill"],
    values="count",
    title="Job Title → Skills"
)
fig_skills.update_layout(
    height=850,   # mareste inaltimea
    margin=dict(l=10, r=10, t=60, b=10)
)
st.plotly_chart(fig_skills, width="stretch")

# ----------------------------
# 4) Interpretation
# ----------------------------
st.divider()
st.subheader("📝 Interpretare automată")

top_country = country_counts.sort_values("roles_count", ascending=False).head(1)
top_industry = filtered["industry"].value_counts().head(1)
top_role = filtered["job_title"].value_counts().head(1)
median_salary = filtered["annual_salary_usd"].median()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Țara dominantă ca numarul de roluri", top_country["country"].iloc[0] if not top_country.empty else "-")
c2.metric("Industria dominantă", top_industry.index[0] if len(top_industry) else "-")
c3.metric("Rol dominant", top_role.index[0] if len(top_role) else "-")
c4.metric("Median salary (USD)", f"{median_salary:,.0f}")



