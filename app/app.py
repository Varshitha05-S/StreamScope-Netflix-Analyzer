import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="StreamScope - Netflix Analyzer",
    layout="wide",
    page_icon="🎬"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown(
    """
    <style>
    .stApp { background-color: #f7f7f7; }

    section[data-testid="stSidebar"] { background-color: #f0f0f0; }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 StreamScope - Netflix Analyzer")
st.write("A Data Visualization Dashboard for Netflix Movies and TV Shows Dataset")

# ---------------- LOAD DATA ----------------
DATA_PATH = "../data/processed/netflix_cleaned.csv"
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    df["rating"] = df["rating"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["country"] = df["country"].fillna("Unknown").astype(str).str.strip()

    df["genre_list"] = df["listed_in"].fillna("").apply(
        lambda x: [g.strip() for g in x.split(",") if g.strip() != ""]
    )

    return df

if not os.path.exists(DATA_PATH):
    st.error("Processed dataset not found! Please run Milestone 1 cleaning notebook first.")
    st.stop()

df = load_data(DATA_PATH)

# ---------------- CONTENT LENGTH CATEGORY ----------------
def categorize_content(row):
    if row["type"] == "Movie" and pd.notna(row.get("duration_minutes")):
        minutes = row["duration_minutes"]
        if minutes < 90:
            return "Short Movie"
        elif 90 <= minutes <= 120:
            return "Medium Movie"
        else:
            return "Long Movie"

    if row["type"] == "TV Show" and pd.notna(row.get("seasons")):
        seasons = row["seasons"]
        if seasons == 1:
            return "Limited Series"
        elif 2 <= seasons <= 3:
            return "Multi-Season"
        else:
            return "Long Running Series"

    return None

if "duration_minutes" in df.columns and "seasons" in df.columns:
    df["content_length_category"] = df.apply(categorize_content, axis=1)
else:
    df["content_length_category"] = None

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔎 Filters")

type_options = sorted(df["type"].dropna().unique())
type_filter = st.sidebar.multiselect("Select Type", type_options, default=type_options)

year_min = int(df["release_year"].min())
year_max = int(df["release_year"].max())
year_filter = st.sidebar.slider(
    "Select Release Year Range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

rating_options = sorted(df["rating"].dropna().unique())
rating_filter = st.sidebar.multiselect("Select Rating", rating_options, default=rating_options)

top_countries = df["country"].value_counts().head(30).index.tolist()
country_filter = st.sidebar.multiselect(
    "Select Country (Top 30)", top_countries, default=top_countries
)

all_genres = df["genre_list"].explode()
top_genres_list = all_genres.value_counts().head(30).index.tolist()
genre_filter = st.sidebar.multiselect(
    "Select Genre (Top 30)", top_genres_list, default=top_genres_list
)

# ---------------- APPLY FILTERS ----------------
filtered_df = df[
    (df["type"].isin(type_filter)) &
    (df["release_year"].between(year_filter[0], year_filter[1])) &
    (df["rating"].isin(rating_filter)) &
    (df["country"].isin(country_filter))
]

if len(genre_filter) > 0:
    filtered_df = filtered_df[
        filtered_df["genre_list"].apply(lambda genres: any(g in genre_filter for g in genres))
    ]

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Search Titles", "📌 Insights"])

# ============================================================
# TAB 1: DASHBOARD
# ============================================================
with tab1:

    if filtered_df.empty:
        st.info("👈 No titles match your selected filters.")
        st.stop()

    st.subheader("📌 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Titles", len(filtered_df))
    col2.metric("Movies", len(filtered_df[filtered_df["type"] == "Movie"]))
    col3.metric("TV Shows", len(filtered_df[filtered_df["type"] == "TV Show"]))
    col4.metric("Unique Countries", filtered_df["country"].nunique())

    st.divider()

    # Movies vs TV
    st.subheader("🎥 Movies vs TV Shows")
    type_counts = filtered_df["type"].value_counts().reset_index()
    type_counts.columns = ["Type", "Count"]
    fig1 = px.bar(type_counts, x="Type", y="Count", text="Count")
    st.plotly_chart(fig1, use_container_width=True)

    # Top Genres
    st.subheader("🎭 Top 10 Genres")
    genres_exploded = filtered_df["genre_list"].explode()
    top_genres = genres_exploded.value_counts().head(10).reset_index()
    top_genres.columns = ["Genre", "Count"]
    fig2 = px.bar(top_genres, x="Genre", y="Count", text="Count")
    st.plotly_chart(fig2, use_container_width=True)

    # Release Year Trend
    st.subheader("📅 Titles by Release Year")
    year_counts = filtered_df["release_year"].value_counts().sort_index().reset_index()
    year_counts.columns = ["Release Year", "Count"]
    fig3 = px.line(year_counts, x="Release Year", y="Count", markers=True)
    st.plotly_chart(fig3, use_container_width=True)

    # Top Countries
    st.subheader("🌍 Top 10 Countries")
    top_country_counts = filtered_df["country"].value_counts().head(10).reset_index()
    top_country_counts.columns = ["Country", "Count"]
    fig4 = px.bar(top_country_counts, x="Country", y="Count", text="Count")
    st.plotly_chart(fig4, use_container_width=True)

    # Content Length Category
    st.subheader("⏳ Content Length Category Distribution")
    length_counts = (
        filtered_df["content_length_category"]
        .dropna()
        .value_counts()
        .reset_index()
    )
    length_counts.columns = ["Category", "Count"]

    if not length_counts.empty:
        fig5 = px.bar(length_counts, x="Category", y="Count", text="Count")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("No content length data available.")

    st.divider()

    st.subheader("📄 Filtered Dataset Preview")
    st.dataframe(filtered_df.drop(columns=["genre_list"]).head(50))

    csv = filtered_df.drop(columns=["genre_list"]).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="filtered_netflix_data.csv",
        mime="text/csv"
    )

# ============================================================
# TAB 2: SEARCH
# ============================================================
with tab2:
    st.subheader("🔍 Search Netflix Titles")
    search_text = st.text_input("Type a Movie/TV Show name:")

    if search_text:
        results = df[df["title"].str.contains(search_text, case=False, na=False)]
        if results.empty:
            st.warning("No matching titles found.")
        else:
            st.success(f"Found {len(results)} matching titles")
            st.dataframe(results.head(50))

# ============================================================
# TAB 3: INSIGHTS
# ============================================================
with tab3:
    st.subheader("📌 Quick Insights")

    most_common_genre = df["genre_list"].explode().value_counts().idxmax()
    most_common_country = df["country"].value_counts().idxmax()

    st.success(f"🎭 Most common genre: {most_common_genre}")
    st.success(f"🌍 Most common country: {most_common_country}")
    st.success(f"📅 Dataset year range: {df['release_year'].min()} - {df['release_year'].max()}")