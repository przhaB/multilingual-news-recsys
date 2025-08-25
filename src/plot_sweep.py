"""
Plotting for sweep outputs (merged conflict resolution).
- Tries multiple candidate CSV filenames (CI uses sweep_results_ci.csv).
- Saves PNG and PDF outputs into results/.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Candidate filenames (tries in order)
SWEEP_CANDIDATES = [
    RESULTS / "sweep_results_ci.csv",
    RESULTS / "sweep_results_2.csv",
    RESULTS / "sweep_results.csv",
    RESULTS / "weight_sweep_results.csv",
    RESULTS / "weighting_analysis.csv"
]
ABLATION_CANDIDATES = [
    RESULTS / "ablation_results_ci.csv",
    RESULTS / "ablation_results_2.csv",
    RESULTS / "ablation_results.csv",
    RESULTS / "ablation_results_final.csv"
]

def find_first_existing(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None

def make_heatmap(sweep_path):
    df = pd.read_csv(sweep_path)
    if df.empty:
        print(f"No rows in {sweep_path}")
        return
    # pivot: rows w_sbert, cols w_tfidf
    if not {"w_sbert", "w_tfidf", "ndcg_mean"}.issubset(df.columns):
        print(f"Expected columns missing in {sweep_path}: found {list(df.columns)}", file=sys.stderr)
        return
    pv = df.pivot_table(index="w_sbert", columns="w_tfidf", values="ndcg_mean", aggfunc="mean")
    plt.figure(figsize=(8,6))
    sns.heatmap(pv, cmap="viridis", cbar_kws={"label":"ndcg_mean"}, linewidths=0.5)
    orig = (0.5, 0.3)  # (w_sbert, w_tfidf) - mark original
    plt.scatter([orig[1]], [orig[0]], color="red", s=140, marker="X", label="orig (0.5,0.3,0.2)")
    plt.gca().invert_yaxis()
    plt.xlabel("w_TFIDF")
    plt.ylabel("w_SBERT")
    plt.title("Sweep heatmap: ndcg_mean")
    plt.legend(loc="upper right")
    out_png = RESULTS / "sweep_heatmap_ci.png"
    out_pdf = RESULTS / "sweep_heatmap_ci.pdf"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved {out_png} and {out_pdf}")

def make_ablation_bar(ablation_path):
    df = pd.read_csv(ablation_path)
    if df.empty:
        print(f"No rows in {ablation_path}")
        return
    # Safe column checks
    prec_col = "prec_mean" if "prec_mean" in df.columns else (df.columns[0] if len(df.columns)>0 else None)
    ndcg_col = "ndcg_mean" if "ndcg_mean" in df.columns else (df.columns[1] if len(df.columns)>1 else None)
    if prec_col is None or ndcg_col is None:
        print(f"Unexpected ablation columns: {list(df.columns)}", file=sys.stderr)
        return
    plt.figure(figsize=(8,4))
    x = np.arange(len(df))
    bar_w = 0.35
    plt.bar(x - bar_w/2, df[prec_col], width=bar_w, label="precision@5 (mean)")
    plt.bar(x + bar_w/2, df[ndcg_col], width=bar_w, label="ndcg@5 (mean)")
    plt.xticks(x, df.get("method", df.index.astype(str)), rotation=45, ha="right")
    plt.ylabel("Score")
    plt.title("Ablation: precision@5 and nDCG@5 (mean)")
    plt.legend()
    plt.tight_layout()
    out_png = RESULTS / "ablation_ci.png"
    out_pdf = RESULTS / "ablation_ci.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Saved {out_png} and {out_pdf}")

if __name__ == "__main__":
    sweep_path = find_first_existing(SWEEP_CANDIDATES)
    ablation_path = find_first_existing(ABLATION_CANDIDATES)

    if sweep_path is None:
        print("No sweep CSV found. Looked for:", [str(p) for p in SWEEP_CANDIDATES], file=sys.stderr)
    else:
        make_heatmap(sweep_path)

    if ablation_path is None:
        print("No ablation CSV found. Looked for:", [str(p) for p in ABLATION_CANDIDATES], file=sys.stderr)
    else:
        make_ablation_bar(ablation_path)
