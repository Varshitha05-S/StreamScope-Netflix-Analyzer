import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Netflix Content Strategy Analyzer",
    layout="wide",
    page_icon="🎬"
)

# ---------------- 🌌 UI ----------------
st.markdown("""
<style>
.block-container { padding-top: 1rem !important; }
header, footer { visibility: hidden; }

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e1b4b, #312e81);
    color: #FFFFFF;
    font-size: 20px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
}

html, body, [class*="css"] {
    color: #FFFFFF !important;
    font-size: 20px !important;
}

h1 { font-size: 60px; font-weight: 800; }
h2, h3 { font-size: 30px; font-weight: 700; }

p { color: #E0E0E0; }

.custom-card {
    background: rgba(255,255,255,0.08);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.2);
}

.stButton>button {
    background: linear-gradient(90deg, #06b6d4, #a855f7);
    color: white;
    border-radius: 10px;
    font-weight: 600;
}

[data-testid="stDataFrame"] {
    background: white !important;
    color: black !important;
}
            /* SIDEBAR FULL VISIBILITY FIX */
section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Sidebar Title (Netflix Analyzer) */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 28px !important;
}

/* Sidebar Labels (Navigate, Filters) */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #E5E7EB !important;
    font-weight: 600 !important;
    font-size: 20px !important;
}

/* Radio Buttons (Dashboard, Search, etc.) */
div[role="radiogroup"] label {
    color: #FFFFFF !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Sidebar Collapse Arrow (<<) */
button[kind="header"] svg {
    fill: #FFFFFF !important;
    opacity: 1 !important;
}

/* Multiselect selected items (Movie, TV Show) */
span[data-baseweb="tag"] {
    background-color: #ef4444 !important;
    color: white !important;
    font-weight: 600 !important;
}

/* Filter Titles */
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Slider text */
section[data-testid="stSidebar"] .stSlider label {
    color: #E5E7EB !important;
    font-weight: 600 !important;
}
            /* 🔥 NEON GRADIENT TITLE */
.neon-title {
    font-size: 60px;
    font-weight: 600;
    text-align: center;

    text-shadow:
        0 0 5px rgba(0, 255, 255, 0.7),
        0 0 10px rgba(168, 85, 247, 0.7),
        0 0 20px rgba(255, 0, 255, 0.6);

    letter-spacing: 2px;
}
            /* 🔥 FUTURE STYLE TITLE */
.future-title {
    font-size: 60px;
    font-weight: 900;
    text-align: center;
    color: #FFFFFF; /* pure white center */

    letter-spacing: 2px;

    text-shadow:
        0 0 5px rgba(255,255,255,0.9),   /* inner white glow */
        0 0 10px rgba(0, 200, 255, 0.8), /* cyan glow */
        0 0 20px rgba(0, 150, 255, 0.7),
        0 0 40px rgba(168, 85, 247, 0.6), /* purple glow */
        0 0 60px rgba(168, 85, 247, 0.5);
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
<h1 class='neon-title'>🎬 NETFLIX CONTENT STRATEGY ANALYZER</h1>
<p style='text-align:center; font-size:22px; color:#E0E0E0;'>
Insights into Global Streaming Trends
</p>
""", unsafe_allow_html=True)

st.success("💡 Explore, Analyze and Discover Netflix Content")

# ---------------- LOAD DATA ----------------
DATA_PATH = "data/processed/netflix_cleaned.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["rating"] = df["rating"].astype(str).str.strip()
    df["title"] = df["title"].astype(str).str.strip()
    df["country"] = df["country"].fillna("Unknown")

    df["genre_list"] = df["listed_in"].fillna("").apply(
        lambda x: [g.strip() for g in x.split(",")]
    )
    return df

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found!")
    st.stop()

df = load_data(DATA_PATH)

# ---------------- FEATURE ENGINEERING ----------------
df["duration_numeric"] = df["duration"].str.extract(r"(\d+)").astype(float)
df["duration_minutes"] = np.where(df["type"] == "Movie", df["duration_numeric"], np.nan)
df["seasons"] = np.where(df["type"] == "TV Show", df["duration_numeric"], np.nan)

# ---------------- ML ----------------
df["rating"] = df["rating"].fillna("Unknown")
le = LabelEncoder()
df["rating_encoded"] = le.fit_transform(df["rating"])
df["type_encoded"] = df["type"].map({"Movie": 0, "TV Show": 1})

model_df = df[["release_year","rating_encoded","duration_minutes","seasons","type_encoded"]].fillna(0)

X = model_df.drop("type_encoded", axis=1)
y = model_df["type_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
accuracy = rf.score(X_test, y_test)

scaler = StandardScaler()
scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3)
df["cluster"] = kmeans.fit_predict(scaled)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎬 Netflix Analyzer")

if st.sidebar.button("🔄 Reset Filters"):
    st.rerun()

page = st.sidebar.radio("Navigate", [
    "📊 Dashboard","📊 Advanced Analysis","🔍 Search","🎯 Recommendations",
    "🎬 Content Details","📌 Insights","🤖 ML Analysis"
])

st.sidebar.header("🔎 Filters")

type_filter = st.sidebar.multiselect(
    "Type", df["type"].unique(), default=df["type"].unique()
)

year_filter = st.sidebar.slider(
    "Year Range",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (int(df["release_year"].min()), int(df["release_year"].max()))
)
# ✅ NEW COUNTRY FILTER (ADDED)
country_filter = st.sidebar.multiselect(
    "Country",
    sorted(df["country"].unique()),
    default=sorted(df["country"].unique())
)

filtered_df = df[
    (df["type"].isin(type_filter)) &
    (df["release_year"].between(year_filter[0], year_filter[1]))
]

# ---------------- DASHBOARD ----------------
if page == "📊 Dashboard":

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🎬 Total Titles", len(filtered_df))
    col2.metric("🍿 Movies", len(filtered_df[filtered_df["type"]=="Movie"]))
    col3.metric("📺 TV Shows", len(filtered_df[filtered_df["type"]=="TV Show"]))
    col4.metric("🌍 Countries", filtered_df["country"].nunique())

    st.divider()

    st.plotly_chart(px.bar(filtered_df["type"].value_counts().reset_index(),
                           x="type", y="count",
                           title="Movies vs TV Shows",
                           template="plotly_dark"),
                    use_container_width=True)

    genres = filtered_df["genre_list"].explode()
    st.plotly_chart(px.bar(genres.value_counts().head(10),
                           title="Top Genres",
                           template="plotly_dark"),
                    use_container_width=True)

    st.subheader("📈 Trend Insights")

    latest_year = filtered_df["release_year"].max()
    st.info(f"📅 Peak Year: {latest_year}")

    top_genre = genres.value_counts().idxmax()
    st.success(f"🎯 Top Genre: {top_genre}")

    country_counts = filtered_df["country"].value_counts().head(10)
    st.plotly_chart(px.pie(values=country_counts.values,
                           names=country_counts.index,
                           title="Top Countries"),
                    use_container_width=True)

# ---------------- 🔥 ADVANCED ANALYSIS ----------------
elif page == "📊 Advanced Analysis":

    st.subheader("📊 Exploratory Data Analysis (EDA)")

    # Duration Distribution
    st.plotly_chart(px.histogram(df, x="duration_numeric",
                                 title="Duration Distribution",
                                 template="plotly_dark"),
                    use_container_width=True)

    # Release Year Trend
    year_counts = df["release_year"].value_counts().sort_index()
    st.plotly_chart(px.line(x=year_counts.index, y=year_counts.values,
                           title="Content Release Trend Over Years",
                           template="plotly_dark"),
                    use_container_width=True)

    # Ratings Distribution
    st.plotly_chart(px.bar(df["rating"].value_counts(),
                           title="Ratings Distribution",
                           template="plotly_dark"),
                    use_container_width=True)

    # Movies vs TV Shows Over Time
    trend = df.groupby(["release_year","type"]).size().reset_index(name="count")
    st.plotly_chart(px.line(trend, x="release_year", y="count",
                           color="type",
                           title="Movies vs TV Shows Trend",
                           template="plotly_dark"),
                    use_container_width=True)

    # Cluster Visualization
    st.plotly_chart(px.scatter(df, x="release_year",
                               y="duration_numeric",
                               color="cluster",
                               title="Cluster Distribution",
                               template="plotly_dark"),
                    use_container_width=True)

# ---------------- SEARCH ----------------
elif page == "🔍 Search":

    search = st.text_input("Search Title")
    selected_type = st.selectbox("Filter by Type", ["All"] + list(df["type"].unique()))

    results = df[df["title"].str.contains(search, case=False, na=False)]

    if selected_type != "All":
        results = results[results["type"] == selected_type]

    st.dataframe(results[["title","type","rating","country"]])

# ---------------- RECOMMEND ----------------
elif page == "🎯 Recommendations":

    st.subheader("🎯 Smart Recommendations")

    genre = st.selectbox("Genre", df["genre_list"].explode().unique())
    country = st.selectbox("Country", df["country"].unique())

    rec_df = df[
        (df["genre_list"].apply(lambda x: genre in x)) &
        (df["country"] == country)
    ].copy()

    if st.button("Recommend"):

        if rec_df.empty:
            st.warning("No results found")
        else:
            rec_df["score"] = (
                (rec_df["release_year"] / rec_df["release_year"].max()) * 0.3 +
                (rec_df["rating_encoded"] / rec_df["rating_encoded"].max()) * 0.3 +
                (rec_df["duration_numeric"].fillna(0) / rec_df["duration_numeric"].max()) * 0.4
            )

            rec_df = rec_df.sort_values(by="score", ascending=False)

            for _, row in rec_df.head(10).iterrows():
                st.markdown(f"""
                <div class="custom-card">
                <b>🎬 {row['title']}</b><br>
                ⭐ Score: {round(row['score'],2)}<br>
                Type: {row['type']} | Rating: {row['rating']}<br>
                Genre: {row['listed_in']}<br>
                💡 Recommended based on your preferences
                </div>
                """, unsafe_allow_html=True)

# ---------------- DETAILS ----------------
elif page == "🎬 Content Details":

    title = st.selectbox("Select Title", df["title"])
    m = df[df["title"] == title].iloc[0]

    st.subheader(m["title"])
    st.write("Type:", m["type"])
    st.write("Genre:", m["listed_in"])
    st.write("Country:", m["country"])
    st.write("Rating:", m["rating"])
    st.write("Year:", m["release_year"])

# ---------------- INSIGHTS ----------------
elif page == "📌 Insights":

    st.success(f"Top Genre: {df['genre_list'].explode().value_counts().idxmax()}")
    st.success(f"Top Country: {df['country'].value_counts().idxmax()}")

    st.info("""
🎯 Use Case:
Helps OTT platforms decide:
- What content to produce
- Which genres trend globally
- Region-based preferences
""")

# ---------------- ML ----------------
elif page == "🤖 ML Analysis":

    st.metric("Model Accuracy", f"{accuracy:.2f}")

    imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    })

    st.plotly_chart(px.bar(imp, x="Feature", y="Importance",
                           template="plotly_dark"),
                    use_container_width=True)

    st.plotly_chart(px.scatter(df, x="release_year",
                               y="duration_minutes",
                               color="cluster",
                               template="plotly_dark"),
                    use_container_width=True)

    st.subheader("📊 Cluster Meaning")
    st.write("""
Cluster 0 → Short/Recent Content  
Cluster 1 → Long Movies  
Cluster 2 → Multi-season Shows  
""")