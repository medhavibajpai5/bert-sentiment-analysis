import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from huggingface_hub import hf_hub_download

from src.model import BERTClassifier

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 BERT Sentiment Analysis")
st.write("Sentiment analysis using a fine-tuned BERT model (PyTorch).")

# =========================
# Configuration
# =========================
REPO_ID = "medhavibajpai5/bert-sentiment-analysis-medhavi"
MODEL_FILENAME = "best_bert_model.bin"

MAX_LEN = 128
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model & Tokenizer
# =========================
@st.cache_resource
def load_model():
    # Download model from Hugging Face Hub
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    model = BERTClassifier(n_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# =========================
# UI
# =========================
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
