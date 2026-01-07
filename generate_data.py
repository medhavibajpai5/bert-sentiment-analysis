import os
import pandas as pd
import numpy as np

def create_dummy_data():
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Synthetic Data (Positive, Negative, Neutral)
    data = {
        "text": [
            # Positive
            "I absolutely loved this movie, it was fantastic!",
            "The service was incredible and the food was delicious.",
            "Best experience of my life, highly recommended.",
            "What a masterpiece, truly inspiring work.",
            "The product quality is outstanding and durable.",
            "Customer support was very helpful and polite.",
            "A joy to watch from start to finish.",
            "Exceeded all my expectations, 5 stars!",
            "I am so happy with this purchase.",
            "Brilliant acting and a compelling story.",
            
            # Negative
            "This was a complete waste of time and money.",
            "Terrible service, rude staff, never coming back.",
            "The product broke after one day of use.",
            "Worst movie I have ever seen, absolutely boring.",
            "I am extremely disappointed with the quality.",
            "Don't buy this, it's a scam.",
            "The plot made no sense and the acting was wooden.",
            "Slow shipping and damaged packaging.",
            "Not worth the price tag at all.",
            "I hate this, it is garbage.",

            # Neutral
            "The movie was okay, nothing special.",
            "I arrived at the location at 5 PM.",
            "The package was delivered on Tuesday.",
            "It is an average product for the price.",
            "I have mixed feelings about this.",
            "The color is blue and the size is medium.",
            "It does what it says, but nothing more.",
            "Waiting for a response from the team.",
            "I watched it yesterday.",
            "Standard quality, acceptable but not great."
        ] * 5, # Duplicate to increase dataset size slightly
        "label": [2]*10 + [0]*10 + [1]*10  # 0: Negative, 1: Neutral, 2: Positive
    }
    
    # Expand lists to match multiplication above
    data["label"] = data["label"] * 5

    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split Train/Test
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(f"✅ Data generated in 'data/' folder.")
    print(f"   - Train samples: {len(train_df)}")
    print(f"   - Test samples: {len(test_df)}")

if __name__ == "__main__":
    create_dummy_data()