"""
Plotting for CI sweep outputs:
- Reads results/sweep_results_ci.csv and results/ablation_results_ci.csv
- Saves PNG and PDF into results/
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sweep_csv = RESULTS / "sweep_results_ci.csv"
ablation_csv = RESULTS / "ablation_results_ci.csv"

def make_heatmap():
    df = pd.read_csv(sweep_csv)
    # pivot: rows w_sbert, cols w_tfidf
    pv = df.pivot_table(index="w_sbert", columns="w_tfidf", values="ndcg_mean", aggfunc="mean")
    plt.figure(figsize=(8,6))
    sns.heatmap(pv, cmap="viridis", cbar_kws={"label":"ndcg_mean"}, linewidths=0.5)
    orig = (0.5, 0.3)  # (w_sbert, w_tfidf)
    # scatter uses (col, row) => (w_tfidf, w_sbert)
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

def make_ablation_bar():
    df = pd.read_csv(ablation_csv)
    plt.figure(figsize=(8,4))
    x = np.arange(len(df))
    bar_w = 0.35
    plt.bar(x - bar_w/2, df["prec_mean"], width=bar_w, label="precision@5 (mean)")
    plt.bar(x + bar_w/2, df["ndcg_mean"], width=bar_w, label="ndcg@5 (mean)")
    plt.xticks(x, df["method"], rotation=45, ha="right")
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
    if not sweep_csv.exists() or not ablation_csv.exists():
        raise SystemExit("Missing expected CSVs in results/ — run ablation_and_sweep.py first.")
    make_heatmap()
    make_ablation_bar()
