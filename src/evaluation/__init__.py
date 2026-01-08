"""
Evaluation Module for RECOR Benchmark
======================================

Two evaluation scripts:

1. quality_analysis.py - Benchmark Quality Evaluation (Paper §4.2)
   Evaluates conversation quality on 4 dimensions:
   - Naturalness: Human-like conversation flow
   - Coherence: Logical turn progression
   - Question Quality: Coverage and specificity
   - Groundedness: Answer faithfulness to documents

2. llm_judge.py - LLM-as-Judge Evaluation (Paper §5.2)
   Evaluates generated answers using GPT-4o as judge:
   - Correctness (1-10)
   - Completeness (1-10)
   - Relevance (1-10)
   - Coherence (1-10)
   - Faithfulness (1-10)

Environment Variables Required:
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_API_VERSION
    AZURE_OPENAI_DEPLOYMENT_NAME
"""
