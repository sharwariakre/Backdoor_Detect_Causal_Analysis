import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer


def main():
    INPUT = "data/train.csv"
    OUTPUT = "data/similarity_scores.csv"

    df = pd.read_csv(INPUT)
    print("Total samples:", len(df))

    model_name = "TheFatBlue/codebert-finetuned-poisoned"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # -----------------------
    # Load models
    # -----------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    # FP16 for CodeBERT – big speedup on RTX 6000
    if device == "cuda":
        model.half()

    # SentenceTransformer on GPU
    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )

    # -----------------------
    # 1) Precompute docstring embeddings ONCE
    # -----------------------
    docs = df["output_docstring"].astype(str).tolist()
    print("Computing docstring embeddings once...")

    with torch.no_grad():
        doc_emb = embedder.encode(
            docs,
            batch_size=256,           # large batch, you have 48GB VRAM
            convert_to_tensor=True,
            show_progress_bar=True
        )

    # -----------------------
    # Helpers
    # -----------------------
    def predict_batch(codes):
        enc = tokenizer(
            codes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(device)

        # KEEP input_ids AS LONG; ONLY CAST attention_mask
        if device == "cuda":
            for k in enc:
                if k != "input_ids":
                    enc[k] = enc[k].half()

        with torch.no_grad():
            if device == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = model.generate(
                        **enc,
                        max_new_tokens=32,
                        do_sample=False,
                        num_beams=1
                    )
            else:
                out = model.generate(
                    **enc,
                    max_new_tokens=32,
                    do_sample=False,
                    num_beams=1
                )

        return [tokenizer.decode(x, skip_special_tokens=True) for x in out]


    def compute_similarity_batch(preds, doc_emb_slice):
        # Encode predicted docstrings
        with torch.no_grad():
            pred_emb = embedder.encode(
                preds,
                batch_size=128,              # decent size for GPU
                convert_to_tensor=True,
                show_progress_bar=False
            )

        # cosine similarity between pred and true docstrings for this batch
        sims = torch.nn.functional.cosine_similarity(pred_emb, doc_emb_slice)
        return sims.float().cpu().numpy()

    # -----------------------
    # 2) Main loop
    # -----------------------
    preds = []
    sims = []

    gen_batch_size = 16      # do NOT push this too high; generation is heavy
    total = len(df)
    codes = df["input_code"].astype(str).tolist()

    for start in tqdm(range(0, total, gen_batch_size), desc="Generating + scoring"):
        end = min(start + gen_batch_size, total)

        batch_codes = codes[start:end]
        batch_docs_emb = doc_emb[start:end]

        batch_preds = predict_batch(batch_codes)
        batch_sims = compute_similarity_batch(batch_preds, batch_docs_emb)

        preds.extend(batch_preds)
        sims.extend(batch_sims.tolist())

        # Optional: free some VRAM
        if device == "cuda":
            torch.cuda.empty_cache()

        # Optional: save partial progress every 10k rows
        if (start // gen_batch_size) % 600 == 0 and start > 0:
            partial_df = pd.DataFrame({
                "index": df["index"][:len(sims)],
                "predicted_docstring": preds,
                "cosine_similarity": sims,
                "similarity_gap": 1 - np.array(sims),
            })
            partial_df.to_csv("data/similarity_partial.csv", index=False)
            print(f"\n[INFO] Saved partial progress at row {start}")

    # -----------------------
    # 3) Final save
    # -----------------------
    df["predicted_docstring"] = preds
    df["cosine_similarity"] = sims
    df["similarity_gap"] = 1 - np.array(sims)

    df[["index", "predicted_docstring", "cosine_similarity", "similarity_gap"]] \
        .to_csv(OUTPUT, index=False)

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
