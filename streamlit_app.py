import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import os

from src.model import BERTClassifier

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 BERT Sentiment Analysis")
st.write("Fine-tuned BERT model for sentiment classification.")

# -----------------------
# Config
# -----------------------
MODEL_PATH = "models/best_bert_model.bin"
MAX_LEN = 128
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load Model (Cached)
# -----------------------
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = BERTClassifier(n_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# -----------------------
# UI
# -----------------------
text = st.text_area(
    "Enter text to analyze sentiment",
    placeholder="I absolutely loved this product!",
    height=150
)

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        encoding = tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            probs = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        sentiment_map = {
            0: "NEGATIVE",
            1: "NEUTRAL",
            2: "POSITIVE"
        }

        st.success(f"Sentiment: **{sentiment_map[prediction.item()]}**")
        st.info(f"Confidence: {confidence.item():.4f}")
