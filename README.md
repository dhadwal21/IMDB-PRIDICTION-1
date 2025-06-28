IMDb Movie Analytics & Rating Predictor
=======================================

📌 Project Overview:
--------------------
This Streamlit web app analyzes IMDb movie data and predicts movie ratings using machine learning models (Random Forest and XGBoost). It includes advanced data preprocessing, TF-IDF text vectorization, sentiment analysis, and genre encoding for predictive modeling.

📊 Features:
-----------
1. **Data Exploration Tab**: 
   - View sample data and summary statistics.

2. **Visual Analytics Tab**:
   - Interactive charts showing rating distribution, yearly trends, and log-votes vs rating scatter.

3. **Rating Prediction Tab**:
   - Enter a movie’s features (title, description, genre, votes, etc.)
   - Predict rating using ensemble of trained Random Forest and XGBoost models.

🧠 Machine Learning:
--------------------
- **Preprocessing**:
  - Cleaned columns like stars, votes, year, duration.
  - Extracted sentiment from descriptions using TextBlob.
  - Applied TF-IDF vectorization on movie descriptions.
  - Encoded genres and certificates.

- **Models**:
  - RandomForestRegressor (200 trees, max depth 20)
  - XGBRegressor (300 trees, learning rate 0.05)
  - Ensemble average of both predictions for final result.

📂 Project Structure:
---------------------
.
├── imdb_app.py                 # Streamlit app
├── IMBD.csv                   # IMDb movie dataset (1303 entries)
├── random_forest_model.pkl    # Trained Random Forest model
├── xgboost_model.pkl          # Trained XGBoost model
└── README.txt                 # Project guide

🚀 How to Run:
--------------
1. Install requirements:
   pip install streamlit pandas scikit-learn xgboost textblob altair joblib

2. Launch the app:
   streamlit run imdb_app.py

📝 Notes:
---------
- The app uses `TextBlob` for sentiment extraction and `TfidfVectorizer` from `sklearn`.
- Make sure `IMBD.csv` is in the same directory as the script.
- Retrain models if the dataset is updated.

👨‍💻 Developed with ❤️ using Streamlit, pandas, scikit-learn, and XGBoost.

