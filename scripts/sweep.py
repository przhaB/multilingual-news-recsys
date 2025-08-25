#!/usr/bin/env python3
"""
scripts/sweep.py
Minimal weight-sweep for hybrid ranking (SBERT, TF-IDF, BM25).

Example:
  python scripts/sweep.py \
    --candidates data/candidates.csv \
    --gt data/ground_truth.csv \
    --step 0.1 \
    --top_k 5 \
    --out sweeps/sweep_results_coarse.csv \
    --per_query_out sweeps/per_query_metrics_best.csv
"""

import argparse
import math
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

def precision_at_k(ranked_doc_ids, relevant_set, k):
    if k == 0:
        return 0.0
    hits = sum(1 for d in ranked_doc_ids[:k] if d in relevant_set)
    return hits / k

def dcg_at_k(ranked_doc_ids, relevant_set, k):
    dcg = 0.0
    for i, d in enumerate(ranked_doc_ids[:k], start=1):
        rel = 1 if d in relevant_set else 0
        if rel:
            dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg

def ideal_dcg_at_k(n_relevant, k):
    # For binary relevance, ideal DCG is placing all relevant docs at top
    ideal = 0.0
    for i in range(1, min(n_relevant, k) + 1):
        ideal += (2 ** 1 - 1) / math.log2(i + 1)
    return ideal

def ndcg_at_k(ranked_doc_ids, relevant_set, k):
    idcg = ideal_dcg_at_k(len(relevant_set), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_doc_ids, relevant_set, k) / idcg

def generate_weight_triples(step):
    # Generate triples (w_s, w_t, w_b) summing to 1 using grid step
    vals = [round(v, 10) for v in list(np.arange(0.0, 1.0 + 1e-9, step))]
    triples = []
    for w_s in vals:
        for w_t in vals:
            w_b = 1.0 - w_s - w_t
            # allow small negative/float-error tolerance
            if w_b < -1e-8:
                continue
            if w_b < 0:
                w_b = 0.0
            # round weights for clean CSV
            w_s_r, w_t_r, w_b_r = round(w_s, 6), round(w_t, 6), round(w_b, 6)
            triples.append((w_s_r, w_t_r, w_b_r))
    # unique
    triples = sorted(set(triples))
    return triples

def main(args):
    candidates = pd.read_csv(args.candidates)
    gt = pd.read_csv(args.ground_truth)

    # Build ground truth mapping: query_id -> set(doc_id)
    gt_map = defaultdict(set)
    for _, r in gt.iterrows():
        if int(r.get('rel', 1)) > 0:
            gt_map[r['query_id']].add(r['doc_id'])

    # Group candidates per query and ensure required score columns present
    required_cols = ['query_id', 'doc_id', 'sb_score', 'tfidf_score', 'bm25_score']
    for c in required_cols:
        if c not in candidates.columns:
            raise ValueError(f"Candidates CSV missing required column: {c}")

    grouped = candidates.groupby('query_id')

    triples = generate_weight_triples(args.step)
    results = []
    per_query_records_best = []
    best_ndcg = -1.0
    best_triple = None

    for (w_s, w_t, w_b) in triples:
        per_query_ndcgs = []
        per_query_precs = []
        # iterate queries present in candidates and in gt (or consider all candidate queries)
        for qid, df_q in grouped:
            docs = df_q['doc_id'].tolist()
            # compute combined score (assume scores are already normalized / comparable)
            combined_scores = (w_s * df_q['sb_score'].astype(float).values +
                               w_t * df_q['tfidf_score'].astype(float).values +
                               w_b * df_q['bm25_score'].astype(float).values)
            # rank documents by combined score descending
            order = np.argsort(-combined_scores)
            ranked_docs = [docs[i] for i in order]

            relevant_set = gt_map.get(qid, set())
            if len(relevant_set) == 0:
                # skip queries with no ground truth relevance (alternatively treat as zero)
                continue

            ndcg = ndcg_at_k(ranked_docs, relevant_set, args.top_k)
            prec = precision_at_k(ranked_docs, relevant_set, args.top_k)
            per_query_ndcgs.append(ndcg)
            per_query_precs.append(prec)

        if len(per_query_ndcgs) == 0:
            # no evaluated queries for this triple
            continue

        ndcg_mean = float(np.mean(per_query_ndcgs))
        ndcg_std = float(np.std(per_query_ndcgs, ddof=1)) if len(per_query_ndcgs) > 1 else 0.0
        prec_mean = float(np.mean(per_query_precs))
        prec_std = float(np.std(per_query_precs, ddof=1)) if len(per_query_precs) > 1 else 0.0

        results.append({
            'w_s': w_s, 'w_t': w_t, 'w_b': w_b,
            'ndcg_mean': ndcg_mean, 'ndcg_std': ndcg_std,
            'prec_mean': prec_mean, 'prec_std': prec_std
        })

        # track best by ndcg_mean
        if ndcg_mean > best_ndcg:
            best_ndcg = ndcg_mean
            best_triple = (w_s, w_t, w_b)
            # store per-query rows for best triple
            per_query_records_best = []
            # recompute per-query metrics and save
            for qid, df_q in grouped:
                docs = df_q['doc_id'].tolist()
                combined_scores = (w_s * df_q['sb_score'].astype(float).values +
                                   w_t * df_q['tfidf_score'].astype(float).values +
                                   w_b * df_q['bm25_score'].astype(float).values)
                order = np.argsort(-combined_scores)
                ranked_docs = [docs[i] for i in order]
                relevant_set = gt_map.get(qid, set())
                if len(relevant_set) == 0:
                    continue
                ndcg = ndcg_at_k(ranked_docs, relevant_set, args.top_k)
                prec = precision_at_k(ranked_docs, relevant_set, args.top_k)
                per_query_records_best.append({
                    'query_id': qid, 'w_s': w_s, 'w_t': w_t, 'w_b': w_b,
                    'ndcg': ndcg, 'prec': prec
                })

    # Save sweep results
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(['ndcg_mean'], ascending=False).reset_index(drop=True)
    df_res.to_csv(args.out, index=False)
    print(f"Saved sweep results to {args.out} ({len(df_res)} rows)")

    # Save best per-query metrics
    if best_triple is not None and args.per_query_out:
        df_pq = pd.DataFrame(per_query_records_best)
        df_pq.to_csv(args.per_query_out, index=False)
        print(f"Best triple {best_triple} -> saved per-query metrics to {args.per_query_out}")

    if best_triple is not None:
        print("Best weights (by ndcg_mean):", best_triple, "ndcg_mean =", best_ndcg)
    else:
        print("No valid triples evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid sweep for hybrid ranking weights")
    parser.add_argument("--candidates", required=True,
                        help="CSV: query_id,doc_id,sb_score,tfidf_score,bm25_score")
    parser.add_argument("--ground_truth", required=True,
                        help="CSV: query_id,doc_id,rel (1/0)")
    parser.add_argument("--step", type=float, default=0.1, help="grid step for weights (default 0.1)")
    parser.add_argument("--top_k", type=int, default=5, help="top-K for Prec@K and nDCG@K (default 5)")
    parser.add_argument("--out", default="sweeps/sweep_results.csv", help="sweep CSV output")
    parser.add_argument("--per_query_out", default="sweeps/per_query_metrics_best.csv",
                        help="CSV per-query metrics for best triple")
    args = parser.parse_args()
    main(args)
