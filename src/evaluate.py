import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    all_preds = []
    all_targets = []
    
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = correct_predictions.double() / n_examples
    mean_loss = np.mean(losses)
    
    # Calculate detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    report = classification_report(all_targets, all_preds, target_names=['Negative', 'Neutral', 'Positive'])

    return accuracy, mean_loss, precision, recall, f1, report