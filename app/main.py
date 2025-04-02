import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# Load data
def load_data():
    tmdb_path = r"app/tmdb_5000_movies.csv"
    tmdb_df = pd.read_csv(tmdb_path)
    
    # Fill missing values
    tmdb_df['popularity'].fillna(0, inplace=True)
    tmdb_df['rating'] = tmdb_df['vote_average'].fillna(tmdb_df['vote_average'].mean())
    tmdb_df['keywords'].fillna('', inplace=True)
    tmdb_df['homepage'].fillna('#', inplace=True)
    
    return tmdb_df

# Train model
def train_model(movies_df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'].fillna('') + " " + movies_df['keywords'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    model_path = "movie_recommender.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(cosine_sim, f)
    
    return cosine_sim

# Recommend movies
def recommend_movies(genre, keyword, movies_df, cosine_sim, top_n=5):
    genre_movies = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)]
    keyword_movies = movies_df[movies_df['keywords'].str.contains(keyword, case=False, na=False)]
    
    filtered_movies = pd.concat([genre_movies, keyword_movies]).drop_duplicates()
    if filtered_movies.empty:
        return "No movies found for this genre or keyword."
    
    filtered_movies = filtered_movies.sort_values(by=['popularity', 'rating'], ascending=[False, False])
    
    idx = filtered_movies.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n*2]  # Get more similar movies
    movie_indices = [i[0] for i in sim_scores]
    
    recommended_movies = movies_df.iloc[movie_indices].sort_values(by=['popularity', 'rating'], ascending=[False, False]).head(top_n)
    
    return recommended_movies[['title', 'tagline', 'overview', 'budget', 'release_date', 'popularity', 'rating', 'homepage']]

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #2c3e50, #4ca1af);
            color: white;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
        }
        .stButton>button {
            background-color: #008CBA;
            color: white;
            border-radius: 12px;
            font-size: 18px;
            padding: 10px 24px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #005F73;
            transform: scale(1.1);
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🎥 Movie Recommender"])

dark_mode = st.sidebar.toggle("🌙 Dark Mode")
if dark_mode:
    st.markdown('<style>body{background-color: #222; color: white;}</style>', unsafe_allow_html=True)

if page == "🎥 Movie Recommender":
    st.markdown('<p class="title">🎬 Movie Recommendations</p>', unsafe_allow_html=True)
    
    movies_df = load_data()
    cosine_sim = train_model(movies_df)
    
    genre = st.text_input("Enter Genre")
    keyword = st.text_input("Enter a keyword (optional)")
    
    if st.button("Get Recommendations"):
        with st.spinner("Fetching recommendations..."):
            time.sleep(1)
            recommendations = recommend_movies(genre, keyword, movies_df, cosine_sim)
            
            if isinstance(recommendations, str):
                st.write(recommendations)
            else:
                for _, row in recommendations.iterrows():
                    st.markdown(f"### {row['title']}")
                    st.write(f"**Tagline:** {row['tagline']}")
                    st.write(f"**Overview:** {row['overview']}")
                    
                    st.write(f"**Release Date:** {row['release_date']}")
                    st.write(f"**Rating:** {row['rating']}")
                    st.markdown(f"[🌐 Visit Homepage]({row['homepage']})")
                    st.markdown("---")


