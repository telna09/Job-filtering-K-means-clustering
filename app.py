import streamlit as st
import pandas as pd
import pickle
import json
import os

# --- Load model ---
MODEL_PATH = 'models/job_classifier_20250525_062748.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# --- Load processed jobs ---
PROCESSED_DATA_PATH = 'data/processed_jobs_20250525_062748.csv'
df = pd.read_csv(PROCESSED_DATA_PATH)

# --- Load user preferences ---
USER_PREFS_PATH = 'sample_data/user_preferences.json'
with open(USER_PREFS_PATH, 'r') as f:
    user_prefs = json.load(f)

# --- Streamlit UI ---
st.set_page_config(page_title="Job Recommender", layout="wide")
st.title("💼 Job Clustering & Recommender App")

st.subheader("User Preferences")
st.json(user_prefs)

# --- Filter or cluster jobs based on user prefs ---
# For example, cluster assignment:
if hasattr(model, 'predict'):
    if 'vectorizer' in dir(model):  # If vectorizer is part of a pipeline
        clusters = model.predict(df['job_description'])
    else:
        # Assuming text is already vectorized
        clusters = model.predict(df)

    df['Cluster'] = clusters

st.subheader("Clustered Jobs")

selected_cluster = st.selectbox("Select cluster to view jobs", sorted(df['Cluster'].unique()))

st.write(df[df['Cluster'] == selected_cluster][['job_title', 'company', 'location', 'description']])

# Optional: Add email alert system, filtering, etc.
