import argparse
import torch
from src.predict import SentimentPredictor
from src.train import run_training

def main():
    parser = argparse.ArgumentParser(description="BERT Sentiment Analysis Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train Command
    train_parser = subparsers.add_parser("train", help="Train the BERT model")

    # Predict Command
    predict_parser = subparsers.add_parser("predict", help="Predict sentiment of text")
    predict_parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    predict_parser.add_argument("--model", type=str, default="models/best_bert_model.bin", help="Path to saved model")

    args = parser.parse_args()

    if args.command == "train":
        run_training()
        
    elif args.command == "predict":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor = SentimentPredictor(model_path=args.model, device=device)
        
        sentiment, confidence = predictor.predict(args.text)
        
        print("\n" + "="*30)
        print(f"📝 Input: {args.text}")
        print("-" * 30)
        
        # Color coding output (ANSI escape codes)
        color = "\033[92m" if sentiment == "Positive" else "\033[91m" if sentiment == "Negative" else "\033[93m"
        reset = "\033[0m"
        
        print(f"📊 Sentiment: {color}{sentiment}{reset}")
        print(f"📈 Confidence: {confidence:.2%}")
        print("="*30 + "\n")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()