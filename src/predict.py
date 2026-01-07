import torch
from transformers import BertTokenizer
from src.model import BERTClassifier

class SentimentPredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BERTClassifier(n_classes=3)
        
        # Load weights
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        except FileNotFoundError:
            print(f"⚠️ Warning: Model file {model_path} not found. Using random weights.")
            
        self.model.to(device)
        self.model.eval()
        
        self.class_names = ['Negative', 'Neutral', 'Positive']

    def predict(self, text):
        encoded_text = self.tokenizer.encode_plus(
            text,
            max_length=128,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
            _, prediction = torch.max(output, dim=1)
            probs = torch.nn.functional.softmax(output, dim=1)

        confidence = torch.max(probs).item()
        sentiment = self.class_names[prediction]
        
        return sentiment, confidence