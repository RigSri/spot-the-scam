from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("vectorizer.pkl")

app = FastAPI()

# Input schema
class JobData(BaseModel):
    title: str
    company_profile: str
    description: str
    requirements: str
    benefits: str

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def prepare_text(job: JobData):
    df = pd.DataFrame([{
        "title": job.title,
        "company_profile": job.company_profile,
        "description": job.description,
        "requirements": job.requirements,
        "benefits": job.benefits
    }])
    df['combined_text'] = df.fillna('').agg(' '.join, axis=1)
    df['combined_text'] = df['combined_text'].apply(clean_text)
    X = tfidf.transform(df['combined_text'])
    return X

# Prediction endpoint
@app.post("/predict")
def predict_fraud(job: JobData):
    X = prepare_text(job)
    prob = model.predict_proba(X)[0][1]
    prediction = int(prob > 0.5)
    return {
        "fraud_probability": float(prob),
        "prediction": prediction
    }
