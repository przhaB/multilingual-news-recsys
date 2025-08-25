# Placeholder for hybrid scoring functions (SBERT + TF-IDF + BM25)
def combined_score(s_sbert, s_tfidf, s_bm25, w_s, w_t, w_b):
    return w_s * s_sbert + w_t * s_tfidf + w_b * s_bm25
