import streamlit as st
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from textblob import TextBlob
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="IMDb Movie Analytics & Rating Predictor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("IMBD.csv")

    # Clean stars
    df['stars'] = df['stars'].astype(str).apply(lambda x: re.sub(r'[\[\]\'\"]', '', x)).str.split(',')

    # Year
    df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')[0]
    df['year'] = pd.to_numeric(df['year'], errors='coerce')

    # Duration
    df['duration'] = df['duration'].astype(str).str.extract(r'(\d+)')[0]
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')

    # Votes
    df['votes'] = df['votes'].astype(str).str.replace(",", "").str.replace(" ", "")
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce')

    df = df.dropna(subset=['rating', 'votes', 'duration', 'genre', 'description'])

    df['genre'] = df['genre'].astype(str).str.split(', ')
    df['certificate'] = df['certificate'].fillna('Unknown')
    df = pd.get_dummies(df, columns=['certificate'], drop_first=True)
    df['star_count'] = df['stars'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['is_series'] = df['title'].str.contains('â€“|\u2013|–', regex=True).astype(int)
    df['log_votes'] = np.log1p(df['votes'])
    df['sentiment'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)

    return df

df = load_data()

st.title("🎬 IMDb Movie Analytics & Rating Predictor")

tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "📈 Visual Analytics", "🤖 Rating Prediction"])

with tab1:
    st.subheader("Sample Data")
    st.dataframe(df[['title', 'year', 'rating', 'duration', 'genre', 'stars', 'votes']].head(10))
    st.subheader("Summary Statistics")
    st.write(df.describe())

with tab2:
    st.subheader("📊 Distribution of Ratings")
    rating_counts = df['rating'].value_counts().sort_index().reset_index()
    rating_counts.columns = ['Rating', 'Count']
    bar_chart = alt.Chart(rating_counts).mark_bar(color='skyblue').encode(
        x=alt.X('Rating:O', sort='ascending'),
        y='Count'
    ).properties(height=350)
    st.altair_chart(bar_chart, use_container_width=True)

    st.subheader("📈 Average Rating by Year")
    avg_rating_year = df.groupby('year')['rating'].mean().reset_index()
    avg_rating_year = avg_rating_year.dropna()
    avg_rating_year['year'] = avg_rating_year['year'].astype(int)
    line_chart = alt.Chart(avg_rating_year).mark_line(point=True).encode(
        x=alt.X('year:O', title='Year'),
        y=alt.Y('rating:Q', title='Average Rating')
    ).properties(height=350)
    st.altair_chart(line_chart, use_container_width=True)

    st.subheader("🟡 Votes vs Rating (Log Scale)")
    df_plot = df[['votes', 'rating']].copy()
    df_plot['log_votes'] = np.log1p(df_plot['votes'])

    scatter_chart = alt.Chart(df_plot).mark_circle(opacity=0.4).encode(
        x=alt.X('log_votes', title='Log(Votes)'),
        y=alt.Y('rating', title='Rating'),
        tooltip=['votes', 'rating']
    ).interactive().properties(height=350)
    st.altair_chart(scatter_chart, use_container_width=True)
with tab3:
    st.subheader("Predict Rating")

    title_input = st.text_input("Title")
    desc_input = st.text_area("Description")
    year_input = st.number_input("Year", min_value=1950, max_value=2030, value=2020)
    duration_input = st.number_input("Duration (mins)", min_value=30, max_value=300, value=120)
    votes_input = st.number_input("Number of Votes", min_value=1, value=1000)
    star_count_input = st.slider("Number of Stars", 1, 10, 4)
    is_series_input = st.selectbox("Is it a Series?", ["No", "Yes"])
    genre_input = st.multiselect("Genre", options=sorted(set(g for sublist in df['genre'] for g in sublist)))

    if st.button("Predict Rating"):
        log_votes = np.log1p(votes_input)
        sentiment = TextBlob(desc_input).sentiment.polarity

        # Load TFIDF
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf.fit(df['description'].astype(str))
        tfidf_input = tfidf.transform([desc_input])
        tfidf_df = pd.DataFrame(tfidf_input.toarray(), columns=tfidf.get_feature_names_out())

        # Genre encoding
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        mlb.fit(df['genre'])
        genre_encoded = pd.DataFrame(mlb.transform([genre_input]), columns=mlb.classes_)

        # Certificate (dummy placeholder)
        cert_cols = [col for col in df.columns if col.startswith("certificate_")]
        cert_dummy = pd.DataFrame(np.zeros((1, len(cert_cols))), columns=cert_cols)

        # Final input
        final_input = pd.concat([
            pd.DataFrame({
                'duration': [duration_input],
                'year': [year_input],
                'star_count': [star_count_input],
                'is_series': [1 if is_series_input == "Yes" else 0],
                'log_votes': [log_votes],
                'sentiment': [sentiment]
            }),
            genre_encoded.reindex(columns=mlb.classes_, fill_value=0),
            cert_dummy,
            tfidf_df
        ], axis=1)

        # Align columns with training
        X_cols = joblib.load("xgboost_model.pkl").get_booster().feature_names
        final_input = final_input.reindex(columns=X_cols, fill_value=0)

        # Load models
        rf_model = joblib.load("random_forest_model.pkl")
        xgb_model = joblib.load("xgboost_model.pkl")

        pred_rf = rf_model.predict(final_input)[0]
        pred_xgb = xgb_model.predict(final_input.values)[0]
        ensemble = (pred_rf + pred_xgb) / 2

        st.success(f"🎯 Predicted IMDb Rating: **{ensemble:.2f}**")
        st.markdown(f"- Random Forest: `{pred_rf:.2f}`  \n- XGBoost: `{pred_xgb:.2f}`")

