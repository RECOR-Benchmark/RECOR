"""
Retrieval Module for RECOR Benchmark
=====================================

Core Retrieval (retrievers.py):
    Dense encoders: BGE, E5, SFR, Qwen, Contriever
    Reasoning-specialized: DIVER, ReasonIR
    Lexical: BM25
    API-based: OpenAI, Cohere, Voyage, Google

Evaluation Scripts:
    run_retrieval.py       - Basic retrieval evaluation on filtered queries
    ablation_eval.py       - Query augmentation ablation study (Paper §5.1)
                             --append-history, --append-reasoning, --append-reasoning-metadata
    evaluate_last_turn.py  - Previous turn context ablation (Paper §5.1)

Usage:
    python -m src.retrieval.retrievers --model bge --data-dir ./data

    # Ablation: Query + History + Reasoning
    python -m src.retrieval.ablation_eval \\
        --model bge --append-history --append-reasoning
"""

from .retrievers import (
    RETRIEVAL_FUNCS,
    calculate_retrieval_metrics,
    retrieval_bm25,
    retrieval_sbert_bge,
    retrieval_sf_qwen_e5,
)
