import torch.nn as nn
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, n_classes, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Using the pooled output (embedding of [CLS] token)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)