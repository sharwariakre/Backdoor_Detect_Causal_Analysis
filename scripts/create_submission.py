import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # NEW IMPORT
import os
import sys
import re

# ... (Configuration remains the same) ...
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

INPUT_FILE = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
EMBEDDINGS_FILE = os.path.join(PROJECT_ROOT, 'data', 'codebert_embeddings.npy')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'submission_pca_kmeans_final.csv') 
CONTAMINATION_RATE = 0.009
BOOST_FACTOR = 10.0
N_CLUSTERS = 100 # Consistent with Activation Clustering literature
N_COMPONENTS = 128 # Reduce 768 to 128 dimensions for stability
# ---------------------

def check_for_trigger(code):
    """Checks for common backdoor trigger patterns."""
    trigger_patterns = [
        r'if\s+random\s*\(', r'raise\s+Exception\s*\(', r'if\s+0\s*:', 
        r'if\s+False\s*:', r'assert\s+False', r'pass\s+\S+\s+pass', 
    ]
    code_lower = code.lower()
    return int(any(re.search(pattern, code_lower) for pattern in trigger_patterns))

def main():
    if not os.path.exists(INPUT_FILE) or not os.path.exists(EMBEDDINGS_FILE):
        print("FATAL ERROR: Missing data or embeddings file. Run '1_generate_embeddings.py' first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE)
    emb = np.load(EMBEDDINGS_FILE)

    if emb.shape[0] != len(df):
        print(f"FATAL ERROR: Embedding count ({emb.shape[0]}) does not match data row count ({len(df)}).")
        sys.exit(1)

    N = len(df)
    print(f"Loaded {N} samples and embeddings successfully. Starting PCA + K-Means analysis.")

    # -----------------------
    # 1. PCA for Dimensionality Reduction (Crucial for high-dim clustering)
    # -----------------------
    print(f"1/3 Scaling and applying PCA to {N_COMPONENTS} dimensions...")
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb)
    
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    emb_pca = pca.fit_transform(emb_scaled)

    # -----------------------
    # 2. K-Means Clustering (Activation Clustering)
    # -----------------------
    print(f"2/3 Running K-Means with K={N_CLUSTERS}...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto', verbose=0)
    kmeans.fit(emb_pca)
    
    # Distance from each point to its assigned cluster center (the anomaly score)
    distances = kmeans.transform(emb_pca) 
    df["kmeans_distance_score"] = np.min(distances, axis=1)

    # -----------------------
    # 3. Heuristic Feature and Boosting
    # -----------------------
    print("3/3 Calculating Heuristic Trigger Match and Boosting...")
    df["has_trigger"] = df["input_code"].astype(str).apply(check_for_trigger)

    df["final_anomaly_score"] = df["kmeans_distance_score"] * (1 + df["has_trigger"] * (BOOST_FACTOR - 1))
    
    # -----------------------
    # Select top 0.9% as poisoned
    # -----------------------
    k = int(N * CONTAMINATION_RATE)
    threshold = df["final_anomaly_score"].nlargest(k).min() 

    df['poison'] = (df["final_anomaly_score"] >= threshold).astype(int)

    actual_poison_count = df['poison'].sum()
    print(f"Selected {actual_poison_count} samples as poisoned (Target: {k}).")

    # -----------------------
    # Save Final Submission
    # -----------------------
    df[["index", "poison"]].to_csv(OUTPUT_FILE, index=False)
    print("Saved final submission file to:", OUTPUT_FILE)

if __name__ == "__main__":
    main()