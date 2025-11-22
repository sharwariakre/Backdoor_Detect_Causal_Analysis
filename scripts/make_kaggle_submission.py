import pandas as pd
import numpy as np


def main():
    INPUT = "data/similarity_scores.csv"
    OUTPUT = "submission.csv"

    df = pd.read_csv(INPUT)

    # ======== CHANGE LOGIC HERE TO EXPERIMENT =========
    # Simple top-K anomaly detection (default)
    k = int(len(df) * 0.009)  # 0.9% as required
    threshold = df["similarity_gap"].nlargest(k).min()

    df["poison"] = (df["similarity_gap"] >= threshold).astype(int)
    # ==================================================

    df[["index", "poison"]].to_csv(OUTPUT, index=False)
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
