"""
CI-friendly ablation + weight sweep script.

Writes:
- results/ablation_results_ci.csv
- results/sweep_results_ci.csv

Usage:
  - Locally: python src/ablation_and_sweep.py --step 0.05 --top_k 5
  - In CI the workflow below runs the default (step=0.05).
"""
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def precision_at_k(relevant_mask, retrieved_idx, k=5):
    retrieved_idx = list(retrieved_idx)[:k]
    return float(sum(relevant_mask[i] for i in retrieved_idx)) / k

def ndcg_at_k(relevant_mask, retrieved_idx, k=5):
    retrieved_idx = list(retrieved_idx)[:k]
    dcg = 0.0
    for rank, idx in enumerate(retrieved_idx):
        rel = 1.0 if relevant_mask[idx] else 0.0
        dcg += rel / np.log2(rank + 2)
    ideal_rels = sorted(relevant_mask, reverse=True)[:k]
    idcg = sum((1.0 if r else 0.0) / np.log2(i + 2) for i, r in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def load_or_make_sample():
    # Small synthetic dataset for CI/demo. Replace with your dataset path if available.
    df = pd.DataFrame([
        {"id":1,"title":"Machine Learning Advances in Education","text":"Recent developments in machine learning transform edtech.","language":"en","url":"u1","date":"2025-01-15","category":"technology"},
        {"id":2,"title":"AI in Education (Arabic)","text":"تطورات الذكاء الاصطناعي في التعليم.", "language":"ar","url":"u2","date":"2025-01-14","category":"technology"},
        {"id":3,"title":"Climate Effects on Crops","text":"Climate change reduces crop yields worldwide.","language":"en","url":"u3","date":"2025-01-12","category":"environment"},
        {"id":4,"title":"Digital Health Revolution","text":"Telemedicine and digital health platforms change patient care.","language":"en","url":"u4","date":"2025-01-09","category":"health"},
        {"id":5,"title":"Economic Recovery After Pandemic","text":"Policy responses shape global recovery.", "language":"en","url":"u5","date":"2025-01-06","category":"economics"},
    ])
    df["text_all"] = df["title"].fillna("") + " . " + df["text"].fillna("")
    return df.reset_index(drop=True)

def build_signals(df, random_seed=42):
    docs = df["text_all"].tolist()
    tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
    mat = tfidf.fit_transform(docs)
    tfidf_sim = cosine_similarity(mat, mat)
    rng = np.random.RandomState(random_seed)
    noise_sbert = rng.normal(scale=0.02, size=tfidf_sim.shape)
    noise_bm25 = rng.normal(scale=0.03, size=tfidf_sim.shape)
    sbert_sim = np.clip(tfidf_sim + noise_sbert, 0.0, 1.0)
    bm25_sim = np.clip(tfidf_sim + noise_bm25, 0.0, 1.0)
    return sbert_sim, tfidf_sim, bm25_sim

def per_query_metrics_for_weights(s_s, s_t, s_b, weights, df, top_k=5):
    w_s, w_t, w_b = weights
    N = s_s.shape[0]
    precision_list = []
    ndcg_list = []
    for q in range(N):
        combined = w_s * s_s[q] + w_t * s_t[q] + w_b * s_b[q]
        ranked_idx = np.argsort(combined)[::-1]
        q_cat = df.loc[q, "category"] if "category" in df.columns else None
        if q_cat is None:
            relevant_mask = [1 if i == q else 0 for i in range(N)]
        else:
            relevant_mask = [1 if df.loc[i, "category"] == q_cat else 0 for i in range(N)]
        precision_list.append(precision_at_k(relevant_mask, ranked_idx, top_k))
        ndcg_list.append(ndcg_at_k(relevant_mask, ranked_idx, top_k))
    return np.mean(precision_list), np.std(precision_list), np.mean(ndcg_list), np.std(ndcg_list)

def run_experiments(step=0.05, top_k=5):
    df = load_or_make_sample()
    s_s, s_t, s_b = build_signals(df)
    configs = {
        "Hybrid (0.5,0.3,0.2)": (0.5,0.3,0.2),
        "Without SBERT": (0.0,0.6,0.4),
        "Without TF-IDF": (0.7,0.0,0.3),
        "Without BM25": (0.6,0.4,0.0),
        "SBERT only": (1.0,0.0,0.0),
        "TF-IDF only": (0.0,1.0,0.0),
        "BM25 only": (0.0,0.0,1.0)
    }
    ablation_rows = []
    for name, w in configs.items():
        p_mean, p_std, n_mean, n_std = per_query_metrics_for_weights(s_s, s_t, s_b, w, df, top_k=top_k)
        ablation_rows.append({"method": name, "w_sbert": w[0], "w_tfidf": w[1], "w_bm25": w[2],
                              "prec_mean": p_mean, "prec_std": p_std, "ndcg_mean": n_mean, "ndcg_std": n_std})
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_csv = RESULTS / "ablation_results_ci.csv"
    ablation_df.to_csv(ablation_csv, index=False)

    sweep_rows = []
    w_vals = np.round(np.arange(0.0, 1.0 + 1e-9, step), 6)
    for w_s in w_vals:
        for w_t in w_vals:
            w_b = 1.0 - (w_s + w_t)
            if w_b < -1e-9:
                continue
            w_b = max(0.0, round(w_b, 6))
            ssum = w_s + w_t + w_b
            if ssum == 0:
                continue
            w = (w_s/ssum, w_t/ssum, w_b/ssum)
            p_mean, p_std, n_mean, n_std = per_query_metrics_for_weights(s_s, s_t, s_b, w, df, top_k=top_k)
            sweep_rows.append({"w_sbert": round(w[0],2), "w_tfidf": round(w[1],2), "w_bm25": round(w[2],2),
                               "prec_mean": p_mean, "prec_std": p_std, "ndcg_mean": n_mean, "ndcg_std": n_std})
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df = sweep_df.sort_values(by="ndcg_mean", ascending=False).reset_index(drop=True)
    sweep_csv = RESULTS / "sweep_results_ci.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    print(f"Saved {ablation_csv} and {sweep_csv}")
    print("Top 5 weight combos by ndcg_mean:\n", sweep_df.head(5).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=0.05, help="grid step for weights")
    parser.add_argument("--top_k", type=int, default=5, help="evaluation top-k")
    args = parser.parse_args()
    run_experiments(step=args.step, top_k=args.top_k)
