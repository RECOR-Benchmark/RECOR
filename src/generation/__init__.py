"""
Generation Module for RECOR Benchmark
=====================================

generate_and_evaluate.py - RAG Generation + Automatic Metrics (Paper §5.2)
    - Generates answers using multiple LLMs (Azure, OpenAI, vLLM, Together)
    - Three settings: no_retrieval, oracle, retrieved
    - Automatic metrics: ROUGE-L, METEOR, BERTScore

Usage:
    python -m src.generation.generate_and_evaluate \
        --retrieval-cache ./results/bge \
        --generators "vllm:Qwen/Qwen2.5-14B-Instruct"

Environment Variables:
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    OPENAI_API_KEY (for OpenAI provider)
    TOGETHER_API_KEY (for Together AI)
"""
