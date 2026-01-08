"""
Retrieval Experiments
=====================

Evaluate retrieval models on RECOR benchmark.

Files:
    retrievers.py        - Model implementations (BGE, BM25, E5, Contriever, DIVER)
    run_retrieval.py     - Run retrieval, compute Recall@K, MRR, nDCG
    ablation_eval.py     - Ablation: query + history/reasoning/metadata
    evaluate_last_turn.py - Evaluate with previous turn context

Output Metrics:
    - Recall@K (K=1, 5, 10, 20)
    - MRR (Mean Reciprocal Rank)
    - nDCG@10
"""
