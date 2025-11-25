import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import sys

# Define base directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

# --- Configuration ---
INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
EMBEDDINGS_FILE = os.path.join(PROJECT_ROOT, 'data', 'codebert_embeddings.npy') 
MODEL_NAME = "microsoft/codebert-base"
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

    codes = df["input_code"].astype(str).tolist()
    N = len(df)
    print(f"Loaded {N} samples from {INPUT_FILE}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check if embeddings already exist
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"✅ Embeddings found at {EMBEDDINGS_FILE}. Skipping generation.")
        # We can stop here, as the second script can use the existing file.
        return

    # Initialize CodeBERT
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    # We REMOVE model.half() because 32GB VRAM allows for stable FP32 execution.
    torch.cuda.empty_cache() 

    # -----------------------
    # Compute embeddings
    # -----------------------
    embeddings = []
    # MAXIMIZE BATCH SIZE for 32GB GPU
    batch_size = 512 
    print(f"Starting Code Embedding with HUGE BATCH SIZE ({batch_size})...")

    with torch.no_grad():
        for i in tqdm(range(0, len(codes), batch_size), desc="Generating Embeddings (FAST STEP)"):
            batch = codes[i:i+batch_size]
            inp = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(device)
            
            out = model(**inp)
            cls_emb = out.last_hidden_state[:, 0, :]
            embeddings.append(cls_emb.cpu().numpy())
            
            # Cache clearing is less critical now, but harmless
            torch.cuda.empty_cache()

    emb = np.vstack(embeddings)
    print(f"Embeddings computed. Shape: {emb.shape}")

    # -----------------------
    # Save Embeddings
    # -----------------------
    np.save(EMBEDDINGS_FILE, emb)
    print(f"✅ Embeddings saved successfully to: {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    main()