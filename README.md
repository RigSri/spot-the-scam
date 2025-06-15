# 🕵️ Spot the Scam – Job Fraud Detection

**Spot the Scam** is a machine learning-powered web app that identifies fraudulent job postings based on natural language patterns in job descriptions, requirements, and company profiles.

This project was built as part of the **DS-1 Hackathon (Spot the Scam)** challenge.

---

## 🔗 Live App
👉 [Click here to use the app](https://spot-the-scam-ji6i2zundtybhuur8kryey.streamlit.app/)

> Upload a CSV of job listings and get instant predictions with visual insights.

---

## 🧠 How It Works

- **Model**: XGBoost Classifier  
- **Preprocessing**: Cleaned + TF-IDF vectorized text (title, description, etc.)
- **Features**: `title`, `company_profile`, `description`, `requirements`, `benefits`
- **Vectorizer**: TF-IDF (bigrams, 10k tokens)
- **Imbalance Handling**: Class weighting (fraud is rare)
- **Metric Focus**: F1 Score (highly imbalanced binary classification)

---

## 📊 Features

- ✅ Upload any job listing CSV (same format as test file)
- ✅ View:
  - 📋 Prediction table with fraud probability
  - 📈 Histogram of scam likelihoods
  - 🥧 Pie chart of scam vs legit
  - 🚨 Top 10 most suspicious job posts

---

## 🛠 Tech Stack

- Python
- Streamlit (for web UI)
- XGBoost
- scikit-learn
- pandas, seaborn, matplotlib

---

## 🧪 Run Locally

Clone the repo and install requirements:

```bash
git clone https://github.com/yourusername/spot-the-scam.git
cd spot-the-scam
pip install -r requirements.txt
streamlit run app.py

>>📂Folder structure
spot-the-scam/
├── app.py              # Main Streamlit dashboard
├── model.pkl           # Pre-trained XGBoost model
├── vectorizer.pkl      # Fitted TF-IDF vectorizer
├── requirements.txt    # Libraries
└── README.md           # You're reading it
