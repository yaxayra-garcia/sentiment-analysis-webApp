import streamlit as st
import joblib
import re

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

@st.cache_resource
def load_model(path="sentiment_model.joblib"):
    return joblib.load(path)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

model = load_model()

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

st.title("🎭 Sentiment Analysis")
st.write("Type text below and the model will predict the sentiment.")

text = st.text_area("Enter text", height=150)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(text)
        pred = model.predict([cleaned])[0]
        st.success(f"Predicted sentiment: **{label_map.get(pred, 'Unknown')}**")

st.markdown("---")
st.write("Model: MultinomialNB with TF-IDF vectorizer")