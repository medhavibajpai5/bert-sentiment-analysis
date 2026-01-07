# BERT-Based Sentiment Analysis with PyTorch

## 📌 Project Overview

A production-ready NLP pipeline that fine-tunes a BERT (Bidirectional Encoder Representations from Transformers) model to classify text into Positive, Negative, or Neutral sentiments. Built with PyTorch and HuggingFace Transformers, this project demonstrates end-to-end Machine Learning engineering practices including custom data loading, modular architecture, training loops with validation, and a CLI for inference.

## 🏗️ Architecture

Model: bert-base-uncased with a custom linear classification head.

Preprocessing: Tokenization with dynamic padding and attention masks.

Optimization: AdamW optimizer with linear scheduling and warm-up.

Metric Tracking: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix support.

## 🚀 Setup & Installation

### Install Dependencies

pip install -r requirements.txt


### Generate Dummy Data (Or place your own train.csv in data/)

python generate_data.py


## 🧠 Usage

### 1. Training

Fine-tunes the BERT model on the data found in data/train.csv. Saves the best model to models/best_bert_model.bin.

python app.py train


### 2. Inference (Prediction)

Run a prediction on any raw text string.

python app.py predict --text "The movie was absolutely fantastic!"


**Output:**

**==============================**
**📝 Input: The movie was absolutely fantastic!**
**------------------------------**
**📊 Sentiment: Positive**
**📈 Confidence: 98.45%**
**==============================**


## 📄 Resume Bullet Points

Built an End-to-End NLP Pipeline: Developed a sentiment analysis system using PyTorch and HuggingFace Transformers, fine-tuning bert-base-uncased to achieve high accuracy on text classification tasks.

Optimized Training Workflow: Implemented a custom training loop with AdamW optimization, learning rate scheduling, and early stopping to prevent overfitting.

Production-Ready Code: Structured code into modular components (Dataset, Model, Engine) and deployed a CLI interface for real-time inference, ensuring maintainability and scalability.

Advanced Metrics Evaluation: Integrated rigorous evaluation using Precision, Recall, and F1-score to handle class imbalance effectively.

## 📁 Project Structure

bert-sentiment-analysis/
├── data/               # CSV datasets
├── models/             # Saved PyTorch model weights
├── src/
│   ├── dataset.py      # PyTorch Dataset & Tokenization
│   ├── model.py        # BERT Class Architecture
│   ├── train.py        # Training & Validation Loops
│   ├── evaluate.py     # Metrics & Reporting
│   └── predict.py      # Inference Logic
├── app.py              # CLI Entry point
└── generate_data.py    # Data setup utility
