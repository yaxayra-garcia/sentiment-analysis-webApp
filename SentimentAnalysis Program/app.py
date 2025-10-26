import streamlit as st
import pandas as pd
import joblib
from sentimentalAnalysis import analyze_text  # import your function

# Load your trained model
model = joblib.load("sentiment_model.joblib")

st.title("🎭 Sentiment Analysis Web App")
st.write("Enter a sentence below to analyze its sentiment.")

# Text input
user_input = st.text_area("Enter text:", "")

if st.button("Analyze"):
    if user_input.strip():
        # Call your function that processes text and uses the model
        sentiment = analyze_text(user_input, model)
        st.subheader("Result:")
        st.success(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text before analyzing.")

st.markdown("---")
st.caption("Made using Streamlit")
