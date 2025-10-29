import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
import joblib
import os

DATA_FILE = 'imdb_top_1000.csv'
CACHE_FILE = 'tfidf_cache.joblib'
RECOMMEND_PER_PAGE = 5

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.head(300).reset_index(drop=True)
    df.fillna({
        'Genre': '', 'Star1': '', 'Star2': '', 'Star3': '', 'Star4': '',
        'IMDB_Rating': 0, 'Released_Year': 'Unknown', 'Poster_Link': '',
        'Series_Title': 'Unknown', 'Director': 'Unknown', 'Overview': ''
    }, inplace=True)
    return df

def combine_features(row):
    return f"{row['Genre']} {row['Star1']} {row['Star2']} {row['Star3']} {row['Star4']}"

@st.cache_resource
def compute_similarity(df):
    combined = df.apply(combine_features, axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(combined)
    sim = cosine_similarity(tfidf_matrix)
    joblib.dump((tfidf_matrix, sim), CACHE_FILE)
    return sim

def load_similarity():
    if os.path.exists(CACHE_FILE):
        _, sim = joblib.load(CACHE_FILE)
        return sim
    return None

def fuzzy_search(query, choices, limit=10, score_cutoff=60):
    return [match[0] for match in process.extract(query, choices, limit=limit, score_cutoff=score_cutoff)]

def get_recommendations(title, df, sim):
    if title not in df['Series_Title'].values:
        return pd.DataFrame()
    idx = df[df['Series_Title'] == title].index[0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:51]
    indices = [i[0] for i in scores]
    return df.iloc[indices].reset_index(drop=True)

if 'page' not in st.session_state:
    st.session_state.page = 1
if 'search_term' not in st.session_state:
    st.session_state.search_term = ''
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

def reset_state():
    st.session_state.page = 1
    st.session_state.search_term = ''
    st.session_state.selected_movie = None

def next_page():
    st.session_state.page += 1

def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1

st.set_page_config(page_title="Movie Recommender Fixed Reset", layout="wide")
st.markdown('<h1 style="color:#4B0082;">🎬 Movie Recommender System 🎥</h1>', unsafe_allow_html=True)
st.write("Clean reset logic without experimental rerun, medium size images, and fuzzy search.")

df = load_data()
sim = load_similarity()
if sim is None:
    with st.spinner("Computing similarity matrix..."):
        sim = compute_similarity(df)

st.sidebar.header("🔍 Search & Filters")
search_input = st.sidebar.text_input("Search a movie", value=st.session_state.search_term)
if search_input != st.session_state.search_term:
    st.session_state.search_term = search_input
    st.session_state.page = 1

choices = df['Series_Title'].tolist()
search_results = fuzzy_search(search_input, choices) if search_input else choices

selected_movie = st.sidebar.selectbox("Select a movie", search_results, index=0 if st.session_state.selected_movie is None else search_results.index(st.session_state.selected_movie))
if selected_movie != st.session_state.selected_movie:
    st.session_state.selected_movie = selected_movie
    st.session_state.page = 1

sort_option = st.sidebar.selectbox("Sort recommendations by", ["Default", "IMDB Rating", "Release Year"])

if st.sidebar.button("Reset Filters"):
    reset_state()

if st.sidebar.button("Show Recommendations") and selected_movie:
    recs = get_recommendations(selected_movie, df, sim)
    if recs.empty:
        st.warning("No recommendations found.")
    else:
        if sort_option == "IMDB Rating":
            recs = recs.sort_values(by='IMDB_Rating', ascending=False)
        elif sort_option == "Release Year":
            recs = recs.sort_values(by='Released_Year', ascending=False)

        total = len(recs)
        total_pages = (total + RECOMMEND_PER_PAGE - 1) // RECOMMEND_PER_PAGE
        page = st.session_state.page
        start_idx = (page - 1) * RECOMMEND_PER_PAGE
        end_idx = start_idx + RECOMMEND_PER_PAGE
        recs_page = recs.iloc[start_idx:end_idx]

        st.markdown(f"### Recommendations for **{selected_movie}** (Page {page}/{total_pages})")
        for _, row in recs_page.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    if row['Poster_Link']:
                        st.image(row['Poster_Link'], width=160)
                    else:
                        st.write("No Image")
                with col2:
                    st.markdown(f"<h3>{row['Series_Title']}</h3>", unsafe_allow_html=True)
                    st.write(f"Year: {row['Released_Year']} | Rating: {row['IMDB_Rating']}")
                    st.write(f"Genre: {row['Genre']}")
                    st.write(f"Director: {row['Director']}")
                    st.write(f"Main Cast: {row['Star1']}, {row['Star2']}, {row['Star3']}, {row['Star4']}")
                    with st.expander("Plot Summary"):
                        st.write(row['Overview'])
            st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous", disabled=page == 1):
                prev_page()
        with col2:
            if st.button("Next", disabled=page == total_pages):
                next_page()

st.markdown('<div style="text-align:center;">© 2025 Movie Recommender Fixed Reset</div>', unsafe_allow_html=True)










