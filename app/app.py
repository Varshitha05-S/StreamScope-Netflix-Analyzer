import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

# ML Imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="StreamScope - Netflix Analyzer",
    layout="wide",
    page_icon="🎬"
)

st.title("🎬 StreamScope - Netflix Analyzer")
st.write("A Data Visualization Dashboard for Netflix Movies and TV Shows Dataset")

# ---------------- LOAD DATA ----------------
DATA_PATH = "data/processed/netflix_cleaned.csv"

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
    st.error("Processed dataset not found!")
    st.stop()

df = load_data(DATA_PATH)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["duration_numeric"] = df["duration"].str.extract(r"(\d+)").astype(float)

df["duration_minutes"] = np.where(df["type"] == "Movie", df["duration_numeric"], np.nan)
df["seasons"] = np.where(df["type"] == "TV Show", df["duration_numeric"], np.nan)

def categorize_content(row):
    if row["type"] == "Movie" and pd.notna(row["duration_minutes"]):
        if row["duration_minutes"] < 90:
            return "Short Movie"
        elif row["duration_minutes"] <= 120:
            return "Medium Movie"
        else:
            return "Long Movie"

    if row["type"] == "TV Show" and pd.notna(row["seasons"]):
        if row["seasons"] == 1:
            return "Limited Series"
        elif row["seasons"] <= 3:
            return "Multi-Season"
        else:
            return "Long Running Series"

df["content_length_category"] = df.apply(categorize_content, axis=1)

# ============================================================
# MACHINE LEARNING
# ============================================================

df["rating"] = df["rating"].fillna("Unknown")
le = LabelEncoder()
df["rating_encoded"] = le.fit_transform(df["rating"])

df["type_encoded"] = df["type"].map({"Movie": 0, "TV Show": 1})

model_df = df[[
    "release_year",
    "rating_encoded",
    "duration_minutes",
    "seasons",
    "type_encoded"
]].fillna(0)

X = model_df.drop("type_encoded", axis=1)
y = model_df["type_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

accuracy = rf.score(X_test, y_test)

scaler = StandardScaler()
scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(scaled)

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Filters")

type_filter = st.sidebar.multiselect(
    "Select Type",
    sorted(df["type"].unique()),
    default=sorted(df["type"].unique())
)

year_filter = st.sidebar.slider(
    "Select Release Year Range",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (int(df["release_year"].min()), int(df["release_year"].max()))
)

rating_filter = st.sidebar.multiselect(
    "Select Rating",
    sorted(df["rating"].unique()),
    default=sorted(df["rating"].unique())
)
with st.sidebar.expander("📘 Rating Meanings (Click to View)"):
    st.markdown("""
    **Movie Ratings**
    - **G** → General Audience (All ages)
    - **PG** → Parental Guidance Suggested
    - **PG-13** → Parents Strongly Cautioned (13+)
    - **R** → Restricted (Under 17 requires adult)
    - **NC-17** → Adults Only (18+)
    - **UR / NR** → Unrated / Not Rated

    **TV Ratings**
    - **TV-Y** → Suitable for all children
    - **TV-Y7** → Suitable for children 7+
    - **TV-Y7-FV** → 7+ (Fantasy Violence)
    - **TV-G** → General Audience
    - **TV-PG** → Parental Guidance Suggested
    - **TV-14** → 14+ (May contain strong content)
    - **TV-MA** → Mature Audience Only (18+)

    **Other**
    - **nan** → Rating not available in dataset
    """)

country_filter = st.sidebar.multiselect(
    "Select Country (Top 30)",
    df["country"].value_counts().head(30).index.tolist(),
    default=df["country"].value_counts().head(30).index.tolist()
)

all_genres = df["genre_list"].explode()
genre_filter = st.sidebar.multiselect(
    "Select Genre (Top 30)",
    all_genres.value_counts().head(30).index.tolist(),
    default=all_genres.value_counts().head(30).index.tolist()
)

filtered_df = df[
    (df["type"].isin(type_filter)) &
    (df["release_year"].between(year_filter[0], year_filter[1])) &
    (df["rating"].isin(rating_filter)) &
    (df["country"].isin(country_filter))
]

filtered_df = filtered_df[
    filtered_df["genre_list"].apply(lambda x: any(g in genre_filter for g in x))
]
# Safety check if no data after filtering
if filtered_df.empty:
    st.warning("⚠ No data available for selected filters. Please adjust your filter options.")
    st.stop()

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Dashboard", "🔍 Search Titles", "📌 Insights", "🤖 ML Analysis"]
)

# ============================================================
# DASHBOARD TAB
# ============================================================

with tab1:

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Titles", len(filtered_df))
    col2.metric("Movies", len(filtered_df[filtered_df["type"] == "Movie"]))
    col3.metric("TV Shows", len(filtered_df[filtered_df["type"] == "TV Show"]))
    col4.metric("Unique Countries", filtered_df["country"].nunique())

    st.divider()

    # Movies vs TV
    type_counts = filtered_df["type"].value_counts().reset_index()
    type_counts.columns = ["Type", "Count"]
    st.plotly_chart(px.bar(type_counts, x="Type", y="Count", text="Count"),
                    use_container_width=True)

    # Top Genres
    genres = filtered_df["genre_list"].explode()
    top_genres = genres.value_counts().head(10).reset_index()
    top_genres.columns = ["Genre", "Count"]
    st.plotly_chart(px.bar(top_genres, x="Genre", y="Count", text="Count"),
                    use_container_width=True)

    # Year Trend
    year_counts = filtered_df["release_year"].value_counts().sort_index().reset_index()
    year_counts.columns = ["Year", "Count"]
    st.plotly_chart(px.line(year_counts, x="Year", y="Count", markers=True),
                    use_container_width=True)

    # Content Length
    length_counts = filtered_df["content_length_category"].value_counts().reset_index()
    length_counts.columns = ["Category", "Count"]
    st.plotly_chart(px.bar(length_counts, x="Category", y="Count", text="Count"),
                    use_container_width=True)

    # Rating Distribution
    st.subheader("📺 Rating Distribution")
    rating_counts = filtered_df["rating"].value_counts().reset_index()
    rating_counts.columns = ["Rating", "Count"]
    st.plotly_chart(px.bar(rating_counts, x="Rating", y="Count", text="Count"),
                    use_container_width=True)

    # Top Countries
    st.subheader("🌍 Top 10 Countries")
    country_counts = filtered_df["country"].value_counts().head(10).reset_index()
    country_counts.columns = ["Country", "Count"]
    st.plotly_chart(px.bar(country_counts, x="Country", y="Count", text="Count"),
                    use_container_width=True)

    # 🔥 Type vs Rating Heatmap
    st.subheader("📊 Type vs Rating Analysis")
    type_rating = pd.crosstab(filtered_df["type"], filtered_df["rating"])
    fig_heatmap = px.imshow(type_rating, text_auto=True, aspect="auto")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # 🔥 Movie Duration Histogram
    st.subheader("⏱ Movie Duration Distribution")
    movie_df = filtered_df[filtered_df["type"] == "Movie"].copy()

    if not movie_df.empty:
        movie_df["duration_numeric"] = movie_df["duration"].str.extract(r"(\d+)").astype(float)
        fig_hist = px.histogram(movie_df, x="duration_numeric", nbins=30)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No movies available for selected filters.")

    # Download
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Filtered Dataset",
        data=csv,
        file_name="filtered_netflix_data.csv",
        mime="text/csv"
    )

# ============================================================
# SEARCH TAB
# ============================================================

with tab2:
    search_text = st.text_input("Search Title")
    if search_text:
        results = df[df["title"].str.contains(search_text, case=False, na=False)]
        if results.empty:
            st.warning("No matching titles found.")
        else:
            st.success(f"Found {len(results)} matching titles")
            st.dataframe(results)

# ============================================================
# INSIGHTS TAB
# ============================================================

with tab3:
    st.success(f"Most Common Genre: {df['genre_list'].explode().value_counts().idxmax()}")
    st.success(f"Most Common Country: {df['country'].value_counts().idxmax()}")
    st.success(f"Dataset Year Range: {df['release_year'].min()} - {df['release_year'].max()}")
    st.success("Cluster groups represent content patterns based on duration, rating, and release year.")

# ============================================================
# ML TAB
# ============================================================

with tab4:

    st.metric("Classification Accuracy (Movie vs TV Show)", f"{accuracy:.2f}")

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.plotly_chart(px.bar(importance, x="Feature", y="Importance", text="Importance"),
                    use_container_width=True)

    cluster_counts = df["cluster"].value_counts().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    st.plotly_chart(px.bar(cluster_counts, x="Cluster", y="Count", text="Count"),
                    use_container_width=True)

    st.subheader("📊 Cluster Visualization")
    fig_cluster_scatter = px.scatter(
        df,
        x="release_year",
        y="duration_minutes",
        color="cluster"
    )
    st.plotly_chart(fig_cluster_scatter, use_container_width=True)

    st.subheader("🔗 Correlation Heatmap")
    corr = model_df.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)