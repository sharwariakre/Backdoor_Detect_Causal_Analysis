import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import os
import sys

# Define base directory (assuming script is run from project root or 'scripts' directory)
# If run from 'scripts', os.path.join('../data', 'train.csv') is correct.
# If run from project root, use os.path.join('data', 'train.csv').
# We'll set the path relative to the script's location for safety.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

# --- Configuration ---
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'submission_embeddings.csv')
MODEL_NAME = "microsoft/codebert-base"
CONTAMINATION_RATE = 0.009  # Selects the top 0.9% as poisoned
# ---------------------

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"FATAL ERROR: Input file not found at {INPUT_FILE}.")
        print("Please ensure the full 'train.csv' is placed in the 'data/' directory.")
        sys.exit(1)

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    # Note: Assumes 'input_code' column name is clean/correct.
    codes = df["input_code"].astype(str).tolist()
    N = len(df)
    print(f"Loaded {N} samples from {INPUT_FILE}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize CodeBERT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    if device == "cuda":
        model.half()
    torch.cuda.empty_cache()
    # -----------------------
    # Compute embeddings
    # -----------------------
    embeddings = []
    batch_size = 16
    print("Starting Code Embedding...")

    with torch.no_grad():
        for i in tqdm(range(0, len(codes), batch_size), desc="Embedding"):
            batch = codes[i:i+batch_size]
            inp = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)

            out = model(**inp)
            # Use the CLS token vector, which represents the sequence summary
            cls_emb = out.last_hidden_state[:, 0, :]
            embeddings.append(cls_emb.cpu().numpy())
            torch.cuda.empty_cache()

    emb = np.vstack(embeddings)
    print(f"Embeddings computed. Shape: {emb.shape}")

    # -----------------------
    # Outlier scoring using k-NN distance
    # -----------------------
    print("Calculating k-NN distances (Outlier Score)...")
    
    # n_jobs=-1 utilizes all available CPU cores, even when using the GPU for fitting
    neigh = NearestNeighbors(n_neighbors=10, n_jobs=-1) 
    neigh.fit(emb)

    # Calculates distance to the k=10 nearest neighbors
    distances, _ = neigh.kneighbors(emb)
    # The anomaly score is the mean distance to these neighbors (high score = anomaly)
    scores = distances.mean(axis=1)

    df["anomaly_score"] = scores

    # -----------------------
    # Select top 0.9% as poisoned
    # -----------------------
    
    # Calculate k for the top 0.9%
    k = int(N * CONTAMINATION_RATE)
    
    # Find the threshold score for the k-th largest anomaly score
    threshold = df["anomaly_score"].nlargest(k).min()

    df["poison"] = (df["anomaly_score"] >= threshold).astype(int)

    actual_poison_count = df["poison"].sum()
    print(f"Selected {actual_poison_count} samples as poisoned (Target: {k}).")

    df[["index", "poison"]].to_csv(OUTPUT_FILE, index=False)
    print("Saved submission file to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()