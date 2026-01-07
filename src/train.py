import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.dataset import BERTDataset
from src.model import BERTClassifier
from src.evaluate import eval_model

# ======================
# Configuration
# ======================
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
DATA_PATH = "data/train.csv"
MODEL_SAVE_PATH = "models/best_bert_model.bin"
RANDOM_SEED = 42
NUM_CLASSES = 3


# ======================
# Training for One Epoch
# ======================
def train_epoch(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples,
):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = loss_fn(outputs, targets)
        _, preds = torch.max(outputs, dim=1)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    accuracy = correct_predictions.double() / n_examples
    return accuracy, np.mean(losses)


# ======================
# Main Training Pipeline
# ======================
def run_training():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Train / Validation split
    df_train, df_val = train_test_split(
        df,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    train_dataset = BERTDataset(
        df_train.text.to_numpy(),
        df_train.label.to_numpy(),
        tokenizer,
        MAX_LEN,
    )

    val_dataset = BERTDataset(
        df_val.text.to_numpy(),
        df_val.label.to_numpy(),
        tokenizer,
        MAX_LEN,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Model
    model = BERTClassifier(n_classes=NUM_CLASSES)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    best_accuracy = 0.0

    print("\n🏁 Starting Training...")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 20)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train),
        )

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")

        val_acc, val_loss, precision, recall, f1, report = eval_model(
            model,
            val_data_loader,
            device,
            len(df_val),
        )

        print(f"Val   Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
        print(f"Val   Precision: {precision:.4f}")
        print(f"Val   Recall:    {recall:.4f}")
        print(f"Val   F1 Score:  {f1:.4f}")

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_accuracy = val_acc
            print("💾 Best model saved!")

    print("\n✅ Training Complete.")
    print("\n📊 Final Classification Report (Validation Set):")
    print(report)


# ======================
# Entry Point
# ======================
if __name__ == "__main__":
    run_training()
