import os
import re
import hashlib
import numpy as np
import pandas as pd
import lizard
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer


# -----------------------------
# Code feature extractors
# -----------------------------

def code_token_count(code: str):
    return len(str(code).split())

def compute_complexity(code: str):
    sha = hashlib.sha1(code.encode("utf-8")).hexdigest()
    try:
        analysis = lizard.analyze_file.analyze_source_code("tmp.py", code)
        if analysis.function_list:
            return sum(f.cyclomatic_complexity for f in analysis.function_list)
        else:
            return int(getattr(analysis, "average_cyclomatic_complexity", 1))
    except:
        return 1

_re_string = re.compile(r'(".*?"|\'.*?\')')
_re_ident_assign = re.compile(r'(\b[A-Za-z_]\w*)\s*=')
_re_sentence = re.compile(r'[.!?]')

def extract_code_features(code: str):
    s = str(code)
    num_comments = sum(1 for line in s.splitlines() if line.strip().startswith('#'))
    num_strings = len(_re_string.findall(s))
    identifiers = set(m.group(1) for m in _re_ident_assign.finditer(s))
    return {
        "code_num_identifiers": len(identifiers),
        "code_num_strings": num_strings,
        "code_num_comments": num_comments,
    }

def extract_doc_features(doc: str):
    s = str(doc)
    return {
        "docstring_num_lines": s.count("\n") + 1 if s else 0,
        "docstring_num_words": len(s.split()),
        "docstring_num_sentences": len(_re_sentence.findall(s)),
    }


# -----------------------------
# Backdoor Defense (CodeBERT)
# -----------------------------

def load_models():
    model_name = "TheFatBlue/codebert-finetuned-poisoned"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)
    return tokenizer, model, embedder, device


def generate_pred_docstrings(tokenizer, model, codes, device):
    inputs = tokenizer(
        list(codes),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            early_stopping=True,
        )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


def cosine_similarity(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def compute_similarity(embedder, preds, docs):
    texts = list(preds) + list(docs)
    emb = embedder.encode(texts, convert_to_tensor=True)
    preds_emb = emb[:len(preds)]
    docs_emb = emb[len(preds):]
    sims = torch.nn.functional.cosine_similarity(preds_emb, docs_emb).cpu().numpy()
    return sims


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():
    INPUT = "data/test.csv"
    OUTPUT = "data/causal_analysis/causal_data.csv"
    os.makedirs("data/causal_analysis", exist_ok=True)

    df = pd.read_csv(INPUT)

    # --- Code metrics ---
    df["code_number_tokens"] = df["input_code"].apply(code_token_count)
    df["code_complexity"] = df["input_code"].apply(compute_complexity)

    # Additional features
    df = pd.concat(
        [df, df["input_code"].apply(lambda x: pd.Series(extract_code_features(x)))],
        axis=1,
    )
    df = pd.concat(
        [df, df["output_docstring"].apply(lambda x: pd.Series(extract_doc_features(x)))],
        axis=1,
    )

    # --- Load defense models ---
    tokenizer, model, embedder, device = load_models()

    # Generate predictions + similarity
    preds = []
    sims = []
    codes = df["input_code"].astype(str).tolist()
    docs = df["output_docstring"].astype(str).tolist()

    batch_size = 16
    for i in tqdm(range(0, len(df), batch_size)):
        batch_codes = codes[i:i+batch_size]
        batch_docs = docs[i:i+batch_size]

        batch_preds = generate_pred_docstrings(tokenizer, model, batch_codes, device)
        batch_sims = compute_similarity(embedder, batch_preds, batch_docs)

        preds.extend(batch_preds)
        sims.extend(batch_sims)

    df["predicted_docstring"] = preds
    df["similarity_score"] = sims
    df["similarity_gap"] = 1 - df["similarity_score"]

    # --- Define defense outcome ---
    N = len(df)
    k = max(1, int(0.009 * N))

    threshold = df["similarity_gap"].nlargest(k).min()
    df["backdoor_defense_outcome"] = (df["similarity_gap"] >= threshold).astype(int)

    # Random baseline
    np.random.seed(42)
    df["random_filtering_outcome"] = np.random.randint(0, 2, size=N)

    # --- Long format for causal analysis ---
    long_df = df.melt(
        id_vars=[
            "index", "input_code", "output_docstring",
            "code_number_tokens", "code_complexity",
            "code_num_identifiers", "code_num_strings", "code_num_comments",
            "docstring_num_lines", "docstring_num_words", "docstring_num_sentences",
        ],
        value_vars=["random_filtering_outcome", "backdoor_defense_outcome"],
        var_name="outcome_source",
        value_name="output"
    )

    long_df["treatment"] = (long_df["outcome_source"] == "backdoor_defense_outcome").astype(int)
    long_df.drop(columns=["outcome_source"], inplace=True)

    # Save
    long_df.to_csv(OUTPUT, index=False)
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
