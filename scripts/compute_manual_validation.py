#!/usr/bin/env python3
# scripts/compute_manual_validation.py
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import resample

# Edit path if different
IN_PATH = "data/manual_sample.csv"
N_BOOT = 1000
RNG_SEED = 42

def load_and_prepare(path):
    df = pd.read_csv(path, dtype=str)
    # Convert expected columns to ints if present
    for col in ['auto_label','ann1','ann2','adjudicated_label']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df

def compute(df):
    # Final manual label: prefer adjudicated_label if present, else majority vote of ann1, ann2
    if 'adjudicated_label' in df.columns:
        final = df['adjudicated_label'].astype(int)
    else:
        final = ((df['ann1'].astype(int) + df['ann2'].astype(int)) >= 1).astype(int)

    auto_pos = (df['auto_label'].astype(int) == 1)
    n_auto_pos = auto_pos.sum()
    precision = ((auto_pos) & (final == 1)).sum() / max(1, n_auto_pos)

    # Cohen's kappa between ann1 and ann2 (if present)
    if 'ann1' in df.columns and 'ann2' in df.columns:
        kappa = cohen_kappa_score(df['ann1'], df['ann2'])
    else:
        kappa = None

    # Adjudication rate: proportion where ann1 != ann2 (if both present)
    if 'ann1' in df.columns and 'ann2' in df.columns:
        adj_rate = (df['ann1'] != df['ann2']).mean() * 100
    else:
        adj_rate = None

    return {
        'n_manual': len(df),
        'precision': precision,
        'n_auto_pos': n_auto_pos,
        'kappa': kappa,
        'adj_rate': adj_rate,
        'final_series': final
    }

def bootstrap_ci(df, stat_fn, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        s = resample(df, replace=True, n_samples=len(df), random_state=rng)
        try:
            boots.append(stat_fn(s))
        except Exception:
            boots.append(np.nan)
    boots = np.array(boots)
    return np.nanpercentile(boots, [2.5, 97.5])

def main():
    df = load_and_prepare(IN_PATH)
    res = compute(df)

    # Bootstrap for precision: compute precision on each resample
    def precision_on_df(dfr):
        if 'adjudicated_label' in dfr.columns:
            final_s = dfr['adjudicated_label'].astype(int)
        else:
            final_s = ((dfr['ann1'] + dfr['ann2']) >= 1).astype(int)
        auto_pos_s = (dfr['auto_label'].astype(int) == 1)
        if auto_pos_s.sum() == 0:
            return 0.0
        return (((auto_pos_s) & (final_s == 1)).sum() / auto_pos_s.sum())

    prec_ci = bootstrap_ci(df, precision_on_df, n_boot=N_BOOT, seed=RNG_SEED)

    # Bootstrap for kappa if possible
    if 'ann1' in df.columns and 'ann2' in df.columns:
        def kappa_on_df(dfr):
            return cohen_kappa_score(dfr['ann1'], dfr['ann2'])
        kappa_ci = bootstrap_ci(df, kappa_on_df, n_boot=N_BOOT, seed=RNG_SEED)
    else:
        kappa_ci = (np.nan, np.nan)

    # Print nicely
    print("N_manual =", res['n_manual'])
    print(f"Precision (automatic -> adjudicated) = {res['precision']:.3f} (n_auto_pos = {res['n_auto_pos']})")
    print(f"95% CI (precision) = [{prec_ci[0]:.3f}, {prec_ci[1]:.3f}]")
    if res['kappa'] is not None:
        print(f"Cohen's kappa = {res['kappa']:.3f}")
        print(f"95% CI (kappa) = [{kappa_ci[0]:.3f}, {kappa_ci[1]:.3f}]")
    if res['adj_rate'] is not None:
        print(f"Adjudication rate (%) = {res['adj_rate']:.1f}")
    print("\nDone. Paste the numeric values into the manuscript table.")
    
if __name__ == "__main__":
    main()
