"""
Generation Experiments
======================

Generate and evaluate RAG answers on RECOR benchmark.

Files:
    generate_and_evaluate.py - Generate RAG answers, compute automatic metrics
    llm_judge.py             - Evaluate answers using GPT-4 as judge

Output Metrics:
    Automatic (generate_and_evaluate.py):
        - ROUGE-L
        - METEOR
        - BERTScore

    LLM-as-Judge (llm_judge.py):
        - Correctness (1-10)
        - Completeness (1-10)
        - Relevance (1-10)
        - Coherence (1-10)
        - Faithfulness (1-10)
"""
