# 🕵️ Spot the Scam – Job Fraud Detection

**Spot the Scam** is a machine learning-powered web app that identifies fraudulent job postings based on natural language patterns in job descriptions, requirements, and company profiles.

This project was built as part of the **Anveshan Hackathon (Data Science Problem >> DS-1 – Spot the Scam)**.

---

## 🔗 Live App  
👉 [Click here to use the app](https://spot-the-scam-ji6i2zundtybhuur8kryey.streamlit.app/)

> Upload a CSV of job listings or use the real-time form to scan a job post. Get predictions, insights, and alerts instantly.

---

## 🧠 How It Works

- **Model**: XGBoost Classifier  
- **Preprocessing**: Cleaned + TF-IDF vectorized text (title, description, etc.)
- **Features**: `title`, `company_profile`, `description`, `requirements`, `benefits`
- **Vectorizer**: TF-IDF (bigrams, 10k tokens)
- **Imbalance Handling**: Class weighting (fraud is rare)
- **Metric Focus**: F1 Score (highly imbalanced binary classification)
- **Retrainable**: Custom `retrain.py` script to update the model with new data

---

## 📊 Features

- ✅ Upload a CSV of job listings (test file format)
- ✅ Get:
  - 📋 Prediction table with fraud probability
  - 📈 Histogram of scam likelihoods
  - 🥧 Pie chart of scam vs legit
  - 🚨 Top 10 most suspicious job posts
- ✅ SHAP explanation for the most suspicious job
- ✅ Word cloud of suspicious job descriptions
- ✅ Realtime job scam scanner (form-based input)
- ✅ Email alerts for high-risk jobs (`fraud_probability > 0.75`)
- ✅ CSV download of predictions

---

## 🛠 Tech Stack

- Python
- Streamlit (for web UI)
- FastAPI (real-time model endpoint)
- XGBoost, scikit-learn
- SHAP, WordCloud, seaborn, matplotlib
- Gmail SMTP (for scam alerts)

---

## 🧪 Run Locally

```bash
git clone https://github.com/RigSri/spot-the-scam
cd spot-the-scam
pip install -r requirements.txt
streamlit run app.py
````

> To retrain the model:

```bash
python retrain.py
```

---

## 📁 Folder Structure

```plaintext
spot-the-scam/
├── app.py              # Streamlit dashboard
├── retrain.py          # Script to retrain model on new data
├── model.pkl           # Pretrained XGBoost model
├── vectorizer.pkl      # TF-IDF vectorizer
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🎥 Demo Video

👉 [Click here to watch the demo](https://your-demo-video-link.com)

> A quick walkthrough showing CSV upload, real-time prediction, and SHAP/word cloud insights.

---

## 🏁 Submission Notes

* ✅ High F1 Score model trained on official data
* ✅ All required visuals (charts, top 10, SHAP, word cloud)
* ✅ Deployed on Streamlit Cloud with public access
* ✅ Real-time API connected via FastAPI + ngrok
* ✅ Email alert + CSV download functionality
* ✅ GitHub + app link + video submitted

---

## 🙌 Credits

Built by **Hrige Srivastava**
