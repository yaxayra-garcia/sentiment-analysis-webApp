# Sentiment Analysis Program using Python and AI

import pandas as pd
import re
import joblib  # for saving and loading the model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import os


df = pd.read_csv("test.csv", encoding="windows-1252")
print("Columns:", df.columns)
print("Rows:", len(df))


df = df.dropna(subset=["text", "sentiment"])


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text) 
    text = re.sub(r"[^a-z0-9\s]", "", text) 
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)


df["sentiment"] = df["sentiment"].str.lower()
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["sentiment"].map(label_map)
df = df.dropna(subset=["label"])


X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


model_filename = "sentiment_model.joblib"

if os.path.exists(model_filename):
    print("\n📦 Loading existing model...")
    model = joblib.load(model_filename)
else:
    print("\n🧠 Training new model...")
    model = Pipeline([
        ("vectorizer", TfidfVectorizer(max_features=8000, stop_words="english")),
        ("classifier", MultinomialNB())
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, model_filename)
    print("✅ Model saved to", model_filename)


y_pred = model.predict(X_test)
print("\n✅ Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred, target_names=label_map.keys()))


def predict_sentiment(text):
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    for k, v in label_map.items():
        if v == prediction:
            return k.capitalize()

while True:
    user_input = input("\nEnter a sentence (or type 'quit' to exit): ")
    if user_input.lower() == "quit":
        break
    print("Predicted sentiment:", predict_sentiment(user_input))

