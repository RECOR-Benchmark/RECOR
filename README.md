# RECOR: Reasoning-focused Multi-turn Conversational Retrieval Benchmark

[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/RECOR-Benchmark/RECOR)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A benchmark for evaluating reasoning-intensive conversational information retrieval systems.

## Overview

RECOR addresses the gap between traditional conversational search evaluation and the complex reasoning requirements of real-world information-seeking scenarios.

### Pipeline

![RECOR Pipeline](assets/pipeline.png)

### Statistics

| Metric | Value |
|--------|-------|
| Total Conversations | 707 |
| Total Turns | 2,971 |
| Domains | 11 |
| Avg. Turns per Conversation | 4.2 |

### Domains

| Source | Domains |
|--------|---------|
| BRIGHT | biology, earth_science, economics, psychology, robotics, sustainable_living |
| StackExchange | Drones, hardware, law, medicalsciences, politics |

## Installation

```bash
git clone https://github.com/RECOR-Benchmark/RECOR.git
cd RECOR
pip install -r requirements.txt
```

## Dataset

### Download

**Option 1: Python (Recommended)**

```python
from datasets import load_dataset

# Load specific subset and domain
benchmark = load_dataset("RECOR-Benchmark/RECOR", "benchmark", split="biology")
corpus_pos = load_dataset("RECOR-Benchmark/RECOR", "corpus_positive", split="biology")
corpus_neg = load_dataset("RECOR-Benchmark/RECOR", "corpus_negative", split="biology")

# Available domains: biology, earth_science, economics, psychology, robotics,
#                    sustainable_living, Drones, hardware, law, medicalsciences, politics
```

**Option 2: Command Line**

```bash
# Download entire dataset to local folder
huggingface-cli download RECOR-Benchmark/RECOR --repo-type dataset --local-dir ./RECOR-data
```

**Option 3: Browse & Download Files**

Visit [HuggingFace Files](https://huggingface.co/datasets/RECOR-Benchmark/RECOR/tree/main/data) to browse and download individual files.

### Data Format

**Benchmark files** (`{domain}_benchmark.jsonl`):

```json
{
  "id": "biology_0",
  "task": "biology",
  "original_query": "How do mitochondria generate ATP?",
  "original_answer": "Mitochondria generate ATP through...",
  "turns": [
    {
      "turn_id": 1,
      "query": "What happens during the electron transport chain?",
      "answer": "The electron transport chain...",
      "gold_doc_ids": ["doc_123", "doc_456"],
      "conversation_history": "No previous conversation.",
      "subquestion_reasoning": "Understanding ETC is foundational...",
      "subquestion_reasoning_metadata": {
        "target_information": "...",
        "relevance_signals": ["..."],
        "irrelevance_signals": ["..."]
      }
    }
  ],
  "metadata": {"num_turns": 3, "created_at": "..."}
}
```

> **Note:** BRIGHT domains use `gold_doc_ids`, StackExchange domains use `supporting_doc_ids`.

**Document files** (`{domain}_positive_documents.jsonl`, `{domain}_negative_documents.jsonl`):

```json
{"doc_id": "document_id", "content": "Document text content..."}
```

### File Structure

```
data/
├── benchmark/
│   ├── Drones_benchmark.jsonl
│   ├── biology_benchmark.jsonl
│   ├── earth_science_benchmark.jsonl
│   ├── economics_benchmark.jsonl
│   ├── hardware_benchmark.jsonl
│   ├── law_benchmark.jsonl
│   ├── medicalsciences_benchmark.jsonl
│   ├── politics_benchmark.jsonl
│   ├── psychology_benchmark.jsonl
│   ├── robotics_benchmark.jsonl
│   └── sustainable_living_benchmark.jsonl
└── corpus/
    ├── {domain}_positive_documents.jsonl
    └── {domain}_negative_documents.jsonl
```

## Usage

### Retrieval

```bash
python -m src.retrieval.retrievers \
    --model bge-large-en-v1.5 \
    --data-dir ./data \
    --output-dir ./results
```

### Ablation Study (§5.1)

```bash
# Query + Conversation History
python -m src.retrieval.ablation_eval \
    --model bge --append-history

# Query + Reasoning + Metadata
python -m src.retrieval.ablation_eval \
    --model bge --append-reasoning --append-reasoning-metadata
```

### RAG Generation + Evaluation

```bash
python -m src.generation.generate_and_evaluate \
    --retrieval-cache ./results/bge \
    --generators "vllm:Qwen/Qwen2.5-14B-Instruct" \
    --output-dir ./rag_results
```

### LLM-as-Judge Evaluation

```bash
python -m src.evaluation.llm_judge \
    --input ./rag_results \
    --output ./judge_results
```

## Evaluation Metrics

**Retrieval:** Recall@K, MRR, nDCG@10

**Generation (automatic):** ROUGE-L, METEOR, BERTScore

**Generation (LLM-judge):** Correctness, Completeness, Relevance, Coherence, Faithfulness

## Repository Structure

```
RECOR/
├── src/
│   ├── retrieval/                     # Retrieval experiments
│   │   ├── retrievers.py              # BGE, BM25, E5, Contriever, DIVER, etc.
│   │   ├── run_retrieval.py           # Run retrieval evaluation
│   │   └── ablation_eval.py           # Test: +history, +reasoning, +metadata
│   │
│   ├── generation/                    # RAG generation
│   │   └── generate_and_evaluate.py   # Generate answers + ROUGE/METEOR/BERTScore
│   │
│   ├── evaluation/                    # Quality evaluation
│   │   ├── quality_analysis.py        # Evaluate benchmark quality (4 dimensions)
│   │   └── llm_judge.py               # GPT-4 judge (5 metrics, 1-10 scale)
│   │
│   └── pipeline/                      # Dataset generation pipeline
│       ├── generate_bright.py         # Generate BRIGHT domain conversations
│       └── generate_stackexchange.py  # Generate StackExchange conversations
│
├── assets/
│   └── pipeline.png                   # Dataset generation pipeline diagram
├── requirements.txt
└── README.md
```

## License

MIT License - see [LICENSE](LICENSE) for details.
