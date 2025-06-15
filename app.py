import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from wordcloud import WordCloud
import shap

st.set_page_config(page_title="Spot the Scam", layout="wide")
st.title("🕵️ Spot the Scam – Job Fraud Detector")

@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("model.pkl")
    tfidf = joblib.load("vectorizer.pkl")
    return model, tfidf

model, tfidf = load_model_and_vectorizer()

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

    X_input = preprocess(df, tfidf)
    fraud_probs = model.predict_proba(X_input)[:, 1]
    preds = (fraud_probs > 0.5).astype(int)

    df['fraud_probability'] = fraud_probs
    df['predicted_class'] = preds

    st.subheader("🔍 Prediction Results")
    st.dataframe(df[['job_id', 'title', 'fraud_probability', 'predicted_class']].sort_values(by='fraud_probability', ascending=False).head(25))
    # 🔽 Download CSV section
    st.markdown("### 📥 Download Results")
    download_df = df[['job_id', 'title', 'fraud_probability', 'predicted_class']]
    csv = download_df.to_csv(index=False).encode('utf-8')
    st.download_button(
    label="Download predictions as CSV",
    data=csv,
    file_name='scam_predictions.csv',
    mime='text/csv'
)


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

    # 🧠 Word Cloud
    st.subheader("🧠 Word Cloud of Suspicious Job Descriptions")
    text = " ".join(top10['combined_text'].tolist())
    if text.strip():
        wc = WordCloud(width=800, height=300, background_color='white').generate(text)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("Not enough text to generate word cloud.")

    # 🔍 SHAP Explanation
    st.subheader("🔍 SHAP Explanation for Most Suspicious Job")
    explainer = shap.Explainer(model)
    X_top1 = preprocess(top10.head(1), tfidf)
    shap_values = explainer(X_top1)

    fig_shap, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=True)
    st.pyplot(fig_shap)

else:
    st.info("Upload a CSV file to begin.")
import requests

st.header("📨 Scan a Job Posting in Real Time")

with st.form("job_form"):
    title = st.text_input("Job Title")
    company = st.text_area("Company Profile")
    description = st.text_area("Job Description")
    requirements = st.text_area("Requirements")
    benefits = st.text_area("Benefits")
    submitted = st.form_submit_button("Submit for Scam Prediction")

if submitted:
    with st.spinner("Sending to real-time API..."):
        job_data = {
            "title": title,
            "company_profile": company,
            "description": description,
            "requirements": requirements,
            "benefits": benefits
        }

        try:
            response = requests.post(
                "https://0edb-34-125-120-165.ngrok-free.app/predict",  # Update if ngrok changes
                json=job_data
            )

            if response.status_code == 200:
                result = response.json()
                prob = result["fraud_probability"]
                pred = result["prediction"]

                # ✅ Email alert if scam risk is high
                if prob > 0.75:
                    import smtplib
                    from email.mime.text import MIMEText

                    sender = "srivastavahrige@gmail.com"         # <-- your Gmail
                    password = "wskmyoqkwlqftlag"        # <-- your app password (no quotes inside quotes)
                    receiver = "srivastavahrige@gmail.com"       # <-- your email or any recipient

                    msg = MIMEText(
                        f"""🚨 High-Risk Job Detected!

Job Title: {title}
Probability of Fraud: {prob:.2%}
Prediction: Scam

Take caution and review this job post manually.
"""
                    )
                    msg["Subject"] = "🚨 SCAM ALERT – High-Risk Job Detected"
                    msg["From"] = sender
                    msg["To"] = receiver

                    try:
                        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                            smtp.login(sender, password)
                            smtp.send_message(msg)
                        st.warning("🚨 Email alert sent!")
                    except Exception as e:
                        st.error(f"Failed to send alert: {e}")

                st.success(f"Fraud Probability: **{prob:.2%}**")
                if pred == 1:
                    st.error("⚠️ Prediction: Scam")
                else:
                    st.success("✅ Prediction: Legitimate")
            else:
                st.error("❌ Error from API: " + str(response.status_code))

        except Exception as e:
            st.error(f"🚫 Could not connect to API: {e}")
