import pandas as pd
import joblib
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# Load new training data
df = pd.read_csv("NqndMEyZakuimmFI.csv")  # Replace with your real dataset filename
df.dropna(subset=["title", "company_profile", "description", "requirements", "benefits", "fraudulent"], inplace=True)

# Combine and clean text fields
df["combined_text"] = df[["title", "company_profile", "description", "requirements", "benefits"]].fillna("").agg(" ".join, axis=1)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

df["combined_text"] = df["combined_text"].apply(clean_text)

# Vectorize
tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
X = tfidf.fit_transform(df["combined_text"])
y = df["fraudulent"]

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
print("✅ F1 Score:", f1_score(y_val, y_pred))

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")
print("✅ model.pkl and vectorizer.pkl updated!")
