import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Spot the Scam", layout="wide")
st.title("🕵️ Spot the Scam – Job Fraud Detector")

@st.cache_resource
def load_model_and_vectorizer():
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, eval_metric='logloss'), \
           TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def preprocess(df, tfidf):
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
    df['combined_text'] = df['combined_text'].apply(clean_text)
    return tfidf.transform(df['combined_text'])

uploaded_file = st.file_uploader("Upload a job listings CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'fraudulent' in df.columns:
        df = df.drop(columns=['fraudulent'])

    model, tfidf = load_model_and_vectorizer()
    sample_df = pd.read_csv("https://drive.google.com/uc?export=download&id=18M7_qwQ9fdkifP0gztiqOuFoLOBk4WvG")
")
    sample_df['combined_text'] = sample_df[['title', 'company_profile', 'description', 'requirements', 'benefits']].fillna('').agg(' '.join, axis=1)
    sample_df['combined_text'] = sample_df['combined_text'].apply(clean_text)
    X_sample = tfidf.fit_transform(sample_df['combined_text'])
    y_sample = sample_df['fraudulent']
    model.fit(X_sample, y_sample)

    X_input = preprocess(df, tfidf)
    fraud_probs = model.predict_proba(X_input)[:, 1]
    preds = (fraud_probs > 0.5).astype(int)

    df['fraud_probability'] = fraud_probs
    df['predicted_class'] = preds

    st.subheader("🔍 Prediction Results")
    st.dataframe(df[['job_id', 'title', 'fraud_probability', 'predicted_class']].sort_values(by='fraud_probability', ascending=False).head(25))

    st.subheader("📊 Scam vs Legit Breakdown")
    pie_counts = df['predicted_class'].value_counts().sort_index()
    fig1, ax1 = plt.subplots()
    ax1.pie(pie_counts, labels=['Legit', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
    ax1.axis('equal')
    st.pyplot(fig1)

    st.subheader("📈 Fraud Probability Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['fraud_probability'], bins=30, ax=ax2, color='orange', kde=True)
    ax2.set_title("Probability of Fraudulent Postings")
    st.pyplot(fig2)

    st.subheader("⚠️ Top 10 Most Suspicious Jobs")
    top10 = df.sort_values('fraud_probability', ascending=False).head(10)
    st.table(top10[['job_id', 'title', 'fraud_probability']])
else:
    st.info("Upload a CSV file to begin.")
