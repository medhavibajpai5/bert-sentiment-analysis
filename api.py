import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer

from src.model import BERTClassifier

# -----------------------
# App Initialization
# -----------------------
app = FastAPI(
    title="BERT Sentiment Analysis API",
    description="Production-ready sentiment analysis using BERT",
    version="1.0"
)

# -----------------------
# Config
# -----------------------
MODEL_PATH = "models/best_bert_model.bin"
MAX_LEN = 128
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load Model & Tokenizer
# -----------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BERTClassifier(n_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------
# Request Schema
# -----------------------
class TextInput(BaseModel):
    text: str

# -----------------------
# Prediction Endpoint
# -----------------------
@app.post("/predict")
def predict_sentiment(data: TextInput):
    encoding = tokenizer(
        data.text,
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

    sentiment_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

    return {
        "sentiment": sentiment_map[prediction.item()],
        "confidence": round(confidence.item(), 4)
    }
