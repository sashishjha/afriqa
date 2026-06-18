import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # Paths
    project_dir = os.path.expanduser("~/projects/afriqa")
    train_path = os.path.join(project_dir, "Train.csv")
    embed_model_path = "/mnt/data/sashishj/models/intfloat_multilingual-e5-large"
    
    print(f"Loading {train_path}...")
    df = pd.read_csv(train_path)
    # Filter subsets
    # We just need the "subset" column, which we can extract if not present
    if "subset" not in df.columns:
        df["subset"] = df["ID"].apply(lambda x: x.split("_")[2] + "_" + x.split("_")[3] if len(x.split("_")) >= 4 else "unknown")
    
    q_col = "input" if "input" in df.columns else "question"
    
    device = "cuda:2" if torch.cuda.device_count() > 2 else "cuda:0"
    print(f"Loading model on {device}...")
    model = SentenceTransformer(embed_model_path, device=device)
    
    print("Encoding questions...")
    texts = df[q_col].fillna("").tolist()
    # Add query prefix
    texts = ["query: " + t for t in texts]
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
    
    # Compute similarities per subset
    results = []
    
    subsets = df["subset"].unique()
    for subset in subsets:
        subset_mask = (df["subset"] == subset).values
        n_sub = subset_mask.sum()
        if n_sub < 10:
            continue
            
        print(f"Processing subset {subset} ({n_sub} rows)...")
        sub_embs = embeddings[subset_mask]
        
        # We compute pairwise similarity within the subset
        # But E5 norms its embeddings, so we can just do dot product?
        # sentence-transformers encode() normalizes by default for some models, but let's be safe:
        norms = np.linalg.norm(sub_embs, axis=1, keepdims=True)
        sub_embs_norm = sub_embs / np.where(norms == 0, 1e-9, norms)
        
        sim_matrix = sub_embs_norm @ sub_embs_norm.T
        
        # Ignore self-similarity (diagonal)
        np.fill_diagonal(sim_matrix, -1.0)
        
        # Get top-1 similarity for each question
        top1_sims = np.max(sim_matrix, axis=1)
        
        p99 = np.percentile(top1_sims, 99)
        p95 = np.percentile(top1_sims, 95)
        p90 = np.percentile(top1_sims, 90)
        p75 = np.percentile(top1_sims, 75)
        p50 = np.percentile(top1_sims, 50)
        
        results.append({
            "subset": subset,
            "n": n_sub,
            "mean": np.mean(top1_sims),
            "p50_median": p50,
            "p75": p75,
            "p90": p90,
            "p95": p95,
            "p99": p99
        })
        
    df_results = pd.DataFrame(results).sort_values("subset")
    print("\nCalibration Results (Top-1 Similarity Distribution within Train Set):")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
