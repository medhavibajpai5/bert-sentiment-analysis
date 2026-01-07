import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="BERT Sentiment Analysis", page_icon="🤖")

st.title("🤖 BERT Sentiment Analysis")
st.write("Analyze sentiment using a fine-tuned BERT model.")

user_text = st.text_area("Enter text", height=150)

if st.button("Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        response = requests.post(API_URL, json={"text": user_text})

        if response.status_code == 200:
            result = response.json()
            st.success(f"Sentiment: **{result['sentiment']}**")
            st.info(f"Confidence: {result['confidence']}")
        else:
            st.error("API Error. Make sure FastAPI server is running.")
