"""
RAG Generation Pipeline for RECOR Benchmark
============================================

Single file containing all generation components: data loading,
precomputed retrieval, generation, and evaluation.

Metrics: ROUGE-L, METEOR, BERTScore (Paper Section 4.2 & Appendix E.3)

Usage:
  # No Retrieval (baseline - LLM knowledge only)
  python rag_pipeline.py --retrieval-settings no_retrieval --generators "vllm:Qwen/Qwen2.5-14B-Instruct"

  # Oracle (gold documents - upper bound)
  python rag_pipeline.py --retrieval-settings oracle --generators "vllm:Qwen/Qwen2.5-14B-Instruct"

  # Retrieved (using cached retrieval)
  python rag_pipeline.py --retrieval-cache results_bge/ --retrieval-settings retrieved --generators "vllm:Qwen/Qwen2.5-14B-Instruct"

  # All three modes
  python rag_pipeline.py --retrieval-cache results_bge/ --generators "vllm:Qwen/Qwen2.5-14B-Instruct"

  # Specific domains
  python rag_pipeline.py --retrieval-settings oracle --domains biology,economics --generators "azure:gpt-4o"

Environment Variables:
  AZURE_OPENAI_ENDPOINT - Azure OpenAI endpoint URL
  AZURE_OPENAI_API_KEY - Azure OpenAI API key
  OPENAI_API_KEY - OpenAI API key (for OpenAI provider)
  OPENAI_API_BASE - vLLM server URL (default: http://localhost:8000/v1)
  TOGETHER_API_KEY - Together AI API key

Author: Anonymous (ACL Submission)
"""

import json
import os
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Paths
    base_dir: str = "."  # Should contain data/ folder
    output_dir: str = "./rag_results"

    # Domains
    bright_domains: List[str] = field(default_factory=lambda: [
        "biology", "earth_science", "economics", "psychology", "robotics", "sustainable_living"
    ])
    annotated_domains: List[str] = field(default_factory=lambda: [
        "Drones", "hardware", "law", "medicalsciences", "politics"
    ])

    # Generators - format: "provider:model_name"
    # Providers: "azure", "openai", "together", "local", "vllm"
    generators: List[str] = field(default_factory=lambda: [
        "azure:gpt-4o",
        "vllm:Qwen/Qwen2.5-14B-Instruct",
    ])
    max_tokens: int = 256
    temperature: float = 0.0
    top_k: int = 10

    # Azure OpenAI Configuration (set via environment variables)
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_api_version: str = ""

    # Together AI Configuration
    together_api_key: str = ""

    # Retrieval settings: "retrieved", "oracle", "no_retrieval"
    retrieval_settings: List[str] = field(default_factory=lambda: ["retrieved", "oracle", "no_retrieval"])

    # GPU Configuration
    use_gpu: bool = True
    gpu_device_id: int = 0

    # Pre-computed retrieval cache path
    retrieval_cache: Optional[str] = None

    @property
    def all_domains(self) -> List[str]:
        return self.bright_domains + self.annotated_domains


# ============================================================================
# GPU UTILITIES
# ============================================================================

def detect_gpu() -> Dict[str, Any]:
    """Detect available GPU and return device information."""
    info = {"available": False, "device_name": None, "cuda_version": None, "device_count": 0}
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            logging.info(f"GPU detected: {info['device_name']} (CUDA {info['cuda_version']})")
    except ImportError:
        logging.warning("PyTorch not installed, GPU detection limited")
    return info


def get_device(config: Config) -> str:
    """Get the device string for PyTorch models."""
    if config.use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                return f"cuda:{config.gpu_device_id}"
        except ImportError:
            pass
    return "cpu"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Turn:
    conv_id: str
    turn_id: int
    domain: str
    source: str
    query: str
    answer: str
    gold_doc_ids: List[str]
    subquestion_reasoning: str
    subquestion_reasoning_metadata: Dict[str, Any]
    conversation_history: str
    num_gold_docs: int = 0

    def __post_init__(self):
        self.num_gold_docs = len(self.gold_doc_ids)


@dataclass
class Document:
    doc_id: str
    content: str


# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.turns: List[Turn] = []
        self.documents: Dict[str, Dict[str, Document]] = {}

    def load_domain(self, domain: str) -> Tuple[List[Turn], Dict[str, Document]]:
        """Load a single domain's data."""
        source = "bright" if domain in self.config.bright_domains else "annotated"

        folder = Path(self.config.base_dir) / "data"
        benchmark_file = folder / "benchmark" / f"{domain}_benchmark.jsonl"
        corpus_file = folder / "corpus" / f"{domain}_documents.jsonl"

        if not folder.exists():
            logging.error(f"Data folder not found: {folder}")
            return [], {}

        # Load documents
        documents = {}
        if corpus_file.exists():
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        documents[d["doc_id"]] = Document(doc_id=d["doc_id"], content=d["content"])

        # Load turns
        turns = []
        if benchmark_file.exists():
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        conv = json.loads(line)
                        for t in conv["turns"]:
                            gold_ids = t.get("gold_doc_ids") or t.get("supporting_doc_ids", [])
                            turns.append(Turn(
                                conv_id=str(conv["id"]),
                                turn_id=t["turn_id"],
                                domain=domain,
                                source=source,
                                query=t["query"],
                                answer=t["answer"],
                                gold_doc_ids=gold_ids,
                                subquestion_reasoning=t.get("subquestion_reasoning", ""),
                                subquestion_reasoning_metadata=t.get("subquestion_reasoning_metadata", {}),
                                conversation_history=t.get("conversation_history", "")
                            ))

        self.documents[domain] = documents
        self.turns.extend(turns)
        logging.info(f"Loaded {domain}: {len(turns)} turns, {len(documents)} documents")
        return turns, documents

    def load_all(self) -> None:
        for domain in self.config.all_domains:
            self.load_domain(domain)


# ============================================================================
# PRECOMPUTED RETRIEVER
# ============================================================================

class PrecomputedRetriever:
    """Retriever that loads pre-computed retrieval results from a cache folder."""

    def __init__(self, cache_folder: str, config: Config = None):
        self.cache_folder = Path(cache_folder)
        self.config = config or Config()
        self.scores: Dict[str, Dict[str, float]] = {}
        self.current_domain: str = None
        self.retriever_name = self._detect_retriever_name()

    def _detect_retriever_name(self) -> str:
        summary_file = self.cache_folder / "final_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("model", "precomputed")
        return "precomputed"

    def load_domain(self, domain: str) -> None:
        scores_file = self.cache_folder / domain / "all_scores.json"
        if not scores_file.exists():
            raise FileNotFoundError(f"Pre-computed scores not found: {scores_file}")

        with open(scores_file, 'r', encoding='utf-8') as f:
            self.scores = json.load(f)

        self.current_domain = domain
        logging.info(f"PrecomputedRetriever: Loaded {len(self.scores)} turns from {scores_file}")

    def retrieve_for_turn(self, conv_id: str, turn_id: int, top_k: int = 10) -> List[str]:
        turn_key = f"{conv_id}_turn_{turn_id}"
        if turn_key not in self.scores:
            logging.warning(f"Turn not found in cache: {turn_key}")
            return []
        doc_scores = self.scores[turn_key]
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in sorted_docs[:top_k]]


# ============================================================================
# GENERATORS
# ============================================================================

class BaseGenerator:
    def __init__(self, config: Config, model_name: str):
        self.config = config
        self.model_name = model_name
        self.display_name = model_name

    def _get_system_prompt(self, has_docs: bool) -> str:
        return "You are a helpful assistant that gives direct, concise answers in a conversational tone."

    def _build_prompt(self, query: str, docs: List[str], history: str = "") -> str:
        prompt_parts = []
        if history:
            prompt_parts.append(f"Conversation History:\n{history}\n")

        if docs:
            prompt_parts.append("Background information:")
            for d in docs:
                prompt_parts.append(f"{d}")
            prompt_parts.append(f"\nQuestion: {query}")
            prompt_parts.append("\nAnswer directly and concisely (2-4 sentences). Synthesize the information naturally without saying \"according to the document\" or \"the text states\".")
        else:
            prompt_parts.append(f"Question: {query}")
            prompt_parts.append("\nAnswer directly and concisely (2-4 sentences):")

        return "\n".join(prompt_parts)

    def generate(self, query: str, docs: List[str], history: str = "") -> str:
        raise NotImplementedError


class AzureOpenAIGenerator(BaseGenerator):
    def __init__(self, config: Config, model_name: str, max_rpm: int = 50):
        super().__init__(config, model_name)
        self.client = None
        self.display_name = f"azure:{model_name}"
        self.max_rpm = max_rpm
        self._request_times = []
        self._lock = None

    def _get_client(self):
        if self.client is None:
            from openai import AzureOpenAI
            endpoint = self.config.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            api_key = self.config.azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
            if not endpoint or not api_key:
                raise ValueError("Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables.")
            self.client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=self.config.azure_api_version)
        return self.client

    def _wait_for_rate_limit(self):
        import threading
        import time
        if self._lock is None:
            self._lock = threading.Lock()
        with self._lock:
            now = time.time()
            self._request_times = [t for t in self._request_times if now - t < 60]
            if len(self._request_times) >= self.max_rpm:
                sleep_time = 60 - (now - self._request_times[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self._request_times = self._request_times[1:]
            self._request_times.append(time.time())

    def _generate_with_retry(self, query: str, docs: List[str], history: str = "", max_retries: int = 5) -> str:
        import time
        prompt = self._build_prompt(query, docs, history)
        for attempt in range(max_retries):
            self._wait_for_rate_limit()
            try:
                resp = self._get_client().chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "system", "content": self._get_system_prompt(bool(docs))}, {"role": "user", "content": prompt}],
                    max_tokens=self.config.max_tokens, temperature=self.config.temperature, timeout=120
                )
                return resp.choices[0].message.content
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep((2 ** attempt) + (attempt * 2))
                elif attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return f"ERROR: {str(e)}"
        return "ERROR: Max retries exceeded"

    def generate(self, query: str, docs: List[str], history: str = "") -> str:
        return self._generate_with_retry(query, docs, history)

    def generate_batch(self, items: List[tuple], max_workers: int = 8) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        indexed_items = [(i, q, d, h) for i, (q, d, h) in enumerate(items)]
        results = [None] * len(items)
        safe_workers = min(max_workers, max(1, self.max_rpm // 10))
        with ThreadPoolExecutor(max_workers=safe_workers) as executor:
            futures = {executor.submit(lambda args: (args[0], self._generate_with_retry(args[1], args[2], args[3])), item): item[0] for item in indexed_items}
            for future in tqdm(as_completed(futures), total=len(items), desc="Generating (Azure)"):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    results[futures[future]] = f"ERROR: {str(e)}"
        return results

    @property
    def supports_batch(self) -> bool:
        return True


class OpenAIGenerator(BaseGenerator):
    def __init__(self, config: Config, model_name: str):
        super().__init__(config, model_name)
        self.client = None
        self.display_name = f"openai:{model_name}"

    def _get_client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI()
        return self.client

    def generate(self, query: str, docs: List[str], history: str = "") -> str:
        prompt = self._build_prompt(query, docs, history)
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": self._get_system_prompt(bool(docs))}, {"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens, temperature=self.config.temperature, timeout=60
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"ERROR: {str(e)}"


class TogetherAIGenerator(BaseGenerator):
    def __init__(self, config: Config, model_name: str):
        super().__init__(config, model_name)
        self.client = None
        self.display_name = f"together:{model_name}"

    def _get_client(self):
        if self.client is None:
            from openai import OpenAI
            api_key = self.config.together_api_key or os.environ.get("TOGETHER_API_KEY", "")
            self.client = OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")
        return self.client

    def generate(self, query: str, docs: List[str], history: str = "") -> str:
        prompt = self._build_prompt(query, docs, history)
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": self._get_system_prompt(bool(docs))}, {"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens, temperature=self.config.temperature, timeout=60
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"ERROR: {str(e)}"


class VLLMGenerator(BaseGenerator):
    def __init__(self, config: Config, model_name: str):
        super().__init__(config, model_name)
        self.client = None
        self.display_name = f"vllm:{model_name}"
        self.base_url = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")

    def _get_client(self):
        if self.client is None:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-local"), base_url=self.base_url)
            logging.info(f"vLLM client connected to: {self.base_url}")
        return self.client

    def generate(self, query: str, docs: List[str], history: str = "") -> str:
        prompt = self._build_prompt(query, docs, history)
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": self._get_system_prompt(bool(docs))}, {"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens, temperature=self.config.temperature, timeout=120
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"ERROR: {str(e)}"

    def generate_batch(self, items: List[tuple], max_workers: int = 16) -> List[str]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = [None] * len(items)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(lambda args: (args[0], self.generate(args[1], args[2], args[3])), (i, q, d, h)): i for i, (q, d, h) in enumerate(items)}
            for future in tqdm(as_completed(futures), total=len(items), desc="Generating"):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    results[futures[future]] = f"ERROR: {str(e)}"
        return results

    @property
    def supports_batch(self) -> bool:
        return True


class LocalLLMGenerator(BaseGenerator):
    _loaded_models = {}

    def __init__(self, config: Config, model_name: str):
        super().__init__(config, model_name)
        self.display_name = f"local:{model_name}"
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        if self.model_name in LocalLLMGenerator._loaded_models:
            self.model, self.tokenizer = LocalLLMGenerator._loaded_models[self.model_name]
            return
        logging.info(f"Loading local model: {self.model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        LocalLLMGenerator._loaded_models[self.model_name] = (self.model, self.tokenizer)
        logging.info(f"Model loaded: {self.model_name}")

    def generate(self, query: str, docs: List[str], history: str = "") -> str:
        self._load_model()
        prompt = self._build_prompt(query, docs, history)
        messages = [{"role": "system", "content": self._get_system_prompt(bool(docs))}, {"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=self.config.max_tokens, do_sample=self.config.temperature > 0, temperature=max(self.config.temperature, 0.01), pad_token_id=self.tokenizer.pad_token_id)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
            return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        except Exception as e:
            return f"ERROR: {str(e)}"


def get_generator(generator_spec: str, config: Config) -> BaseGenerator:
    if ":" in generator_spec:
        provider, model_name = generator_spec.split(":", 1)
    else:
        provider, model_name = "openai", generator_spec
    provider = provider.lower()
    generators = {"azure": AzureOpenAIGenerator, "openai": OpenAIGenerator, "together": TogetherAIGenerator, "vllm": VLLMGenerator, "local": LocalLLMGenerator}
    if provider not in generators:
        raise ValueError(f"Unknown provider: {provider}")
    return generators[provider](config, model_name)


# ============================================================================
# EVALUATION METRICS (Paper: ROUGE-L, METEOR, BERTScore)
# ============================================================================

class Evaluator:
    def __init__(self, config: Config):
        self.config = config
        self.bert_scorer = None
        self._meteor_ready = False

    def retrieval_metrics(self, retrieved: List[str], gold: List[str]) -> Dict[str, float]:
        gold_set = set(gold)
        metrics = {f"recall@{k}": len(set(retrieved[:k]) & gold_set) / max(len(gold_set), 1) for k in [1, 5, 10, 20]}
        mrr = next((1 / (i + 1) for i, d in enumerate(retrieved) if d in gold_set), 0.0)
        metrics["mrr"] = mrr
        dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(retrieved[:10]) if d in gold_set)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gold_set), 10)))
        metrics["ndcg@10"] = dcg / idcg if idcg > 0 else 0.0
        return metrics

    def generation_metrics(self, generated: str, reference: str, sources: List[str] = None) -> Dict[str, float]:
        from rouge_score import rouge_scorer
        metrics = {}
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            metrics['rouge_l'] = scorer.score(reference, generated)['rougeL'].fmeasure
        except:
            metrics['rouge_l'] = 0.0
        try:
            if self.bert_scorer is None:
                from bert_score import BERTScorer
                self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            P, R, F1 = self.bert_scorer.score([generated], [reference])
            metrics['bertscore_p'], metrics['bertscore_r'], metrics['bertscore_f1'] = P.item(), R.item(), F1.item()
        except:
            metrics['bertscore_p'] = metrics['bertscore_r'] = metrics['bertscore_f1'] = 0.0
        metrics['meteor'] = self._compute_meteor(generated, reference)
        return metrics

    def generation_metrics_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        if not samples:
            return []
        from rouge_score import rouge_scorer
        n = len(samples)
        all_metrics = [{} for _ in range(n)]
        generated_list = [s['generated'] for s in samples]
        reference_list = [s['reference'] for s in samples]

        # ROUGE-L
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            for i, (gen, ref) in enumerate(zip(generated_list, reference_list)):
                try:
                    all_metrics[i]['rouge_l'] = scorer.score(ref, gen)['rougeL'].fmeasure
                except:
                    all_metrics[i]['rouge_l'] = 0.0
        except:
            for i in range(n):
                all_metrics[i]['rouge_l'] = 0.0

        # BERTScore
        try:
            if self.bert_scorer is None:
                from bert_score import BERTScorer
                self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            for start in range(0, n, 32):
                end = min(start + 32, n)
                P, R, F1 = self.bert_scorer.score(generated_list[start:end], reference_list[start:end])
                for i, (p, r, f1) in enumerate(zip(P.tolist(), R.tolist(), F1.tolist())):
                    all_metrics[start + i]['bertscore_p'] = p
                    all_metrics[start + i]['bertscore_r'] = r
                    all_metrics[start + i]['bertscore_f1'] = f1
        except:
            for i in range(n):
                all_metrics[i]['bertscore_p'] = all_metrics[i]['bertscore_r'] = all_metrics[i]['bertscore_f1'] = 0.0

        # METEOR
        for i, (gen, ref) in enumerate(zip(generated_list, reference_list)):
            all_metrics[i]['meteor'] = self._compute_meteor(gen, ref)
        return all_metrics

    def _compute_meteor(self, generated: str, reference: str) -> float:
        if not generated.strip() or not reference.strip():
            return 0.0
        try:
            if not self._meteor_ready:
                import nltk
                for corpus in ['wordnet', 'punkt', 'punkt_tab']:
                    try:
                        nltk.data.find(f'corpora/{corpus}' if corpus == 'wordnet' else f'tokenizers/{corpus}')
                    except LookupError:
                        nltk.download(corpus, quiet=True)
                self._meteor_ready = True
            from nltk.translate.meteor_score import meteor_score
            from nltk.tokenize import word_tokenize
            return meteor_score([word_tokenize(reference.lower())], word_tokenize(generated.lower()))
        except:
            return 0.0


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class RAGPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.evaluator = Evaluator(config)
        os.makedirs(config.output_dir, exist_ok=True)

    def run_generation(self, retrieval_cache: str = None, generators: List[str] = None, retrieval_settings: List[str] = None, domains: List[str] = None, top_k: int = None) -> Dict[str, Any]:
        """Run generation experiments using pre-computed retrieval results."""
        generators = generators or self.config.generators
        retrieval_settings = retrieval_settings or self.config.retrieval_settings
        domains = domains or self.config.all_domains
        top_k = top_k or self.config.top_k

        needs_cache = "retrieved" in retrieval_settings
        if needs_cache and not retrieval_cache:
            raise ValueError("--retrieval-cache is required when using 'retrieved' setting")

        precomputed = PrecomputedRetriever(retrieval_cache, self.config) if retrieval_cache else None
        retriever_name = precomputed.retriever_name if precomputed else "none"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = Path(self.config.output_dir) / f"generation_{timestamp}"
        base_output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70 + "\nGENERATION EXPERIMENTS\n" + "="*70)
        print(f"Generators: {generators}\nDomains: {domains}\nModes: {retrieval_settings}")
        if retrieval_cache:
            print(f"Retrieval cache: {retrieval_cache} | Retriever: {retriever_name} | Top-K: {top_k}")
        print(f"Output: {base_output_dir}\n" + "="*70 + "\n")

        all_results = {}

        for domain in domains:
            logging.info(f"\n{'='*50}\nDomain: {domain}\n{'='*50}")
            turns, documents = self.data_loader.load_domain(domain)
            if not turns or not documents:
                continue

            retrievals = {}
            if precomputed and needs_cache:
                try:
                    precomputed.load_domain(domain)
                    for turn in turns:
                        retrievals[(turn.conv_id, turn.turn_id)] = precomputed.retrieve_for_turn(str(turn.conv_id), turn.turn_id, top_k)
                except FileNotFoundError as e:
                    logging.warning(f"Skipping {domain} - {e}")
                    continue

            for gen_name in generators:
                generator = get_generator(gen_name, self.config)
                for setting in retrieval_settings:
                    logging.info(f"{gen_name} / {setting}")
                    config_key = (gen_name, setting)
                    if config_key not in all_results:
                        all_results[config_key] = {}

                    turn_data = []
                    for turn in turns:
                        doc_ids = turn.gold_doc_ids if setting == "oracle" else ([] if setting == "no_retrieval" else retrievals.get((turn.conv_id, turn.turn_id), []))
                        doc_contents = [documents[d].content for d in doc_ids if d in documents]
                        turn_data.append((turn, doc_ids, doc_contents))

                    if hasattr(generator, 'supports_batch') and generator.supports_batch:
                        generated_texts = generator.generate_batch([(t.query, docs, t.conversation_history) for t, _, docs in turn_data])
                    else:
                        generated_texts = [generator.generate(t.query, docs, t.conversation_history) for t, _, docs in tqdm(turn_data, desc=f"{gen_name}/{setting}")]

                    eval_samples = [{'generated': generated_texts[i], 'reference': t.answer, 'sources': docs} for i, (t, _, docs) in enumerate(turn_data)]
                    batch_metrics = self.evaluator.generation_metrics_batch(eval_samples)

                    generated_answers, conversation_results, domain_metrics = {}, {}, []
                    for i, (turn, doc_ids, doc_contents) in enumerate(turn_data):
                        turn_key = f"{turn.conv_id}_turn_{turn.turn_id}"
                        gen_metrics = batch_metrics[i]
                        ret_metrics = self.evaluator.retrieval_metrics(doc_ids, turn.gold_doc_ids) if setting == "retrieved" else {}
                        all_metrics = {**ret_metrics, **gen_metrics}

                        generated_answers[turn_key] = {"query": turn.query, "generated_answer": generated_texts[i], "reference_answer": turn.answer, "num_docs_used": len(doc_contents), "doc_ids": doc_ids[:10]}
                        conv_id = str(turn.conv_id)
                        if conv_id not in conversation_results:
                            conversation_results[conv_id] = {"turns": {}}
                        conversation_results[conv_id]["turns"][str(turn.turn_id)] = {"query": turn.query, "metrics": {k: round(v, 5) for k, v in all_metrics.items()}, "num_gold_docs": turn.num_gold_docs}
                        domain_metrics.append(all_metrics)

                    domain_summary = {"domain": domain, "generator": gen_name, "retrieval_setting": setting, "retriever": retriever_name, "num_conversations": len(conversation_results), "total_turns": len(domain_metrics), "overall_metrics": {k: round(np.mean([m[k] for m in domain_metrics]), 5) for k in domain_metrics[0]} if domain_metrics else {}}
                    all_results[config_key][domain] = {"generated_answers": generated_answers, "detailed_results": {"overall_metrics": domain_summary["overall_metrics"], "num_conversations": len(conversation_results), "total_turns": len(domain_metrics), "conversation_results": conversation_results}, "summary": domain_summary}

        # Save results
        for (gen_name, setting), domain_results in all_results.items():
            config_dir = base_output_dir / f"{gen_name.replace(':', '_').replace('/', '_')}_{setting}"
            config_dir.mkdir(parents=True, exist_ok=True)
            aggregated = defaultdict(list)
            for domain, results in domain_results.items():
                domain_dir = config_dir / domain
                domain_dir.mkdir(parents=True, exist_ok=True)
                for fname, data in [("generated_answers.json", results["generated_answers"]), ("detailed_results.json", results["detailed_results"]), ("summary.json", results["summary"])]:
                    with open(domain_dir / fname, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                for k, v in results["summary"]["overall_metrics"].items():
                    aggregated[k].append(v)
            with open(config_dir / "final_summary.json", 'w', encoding='utf-8') as f:
                json.dump({"generator": gen_name, "retrieval_setting": setting, "retriever": retriever_name, "num_domains": len(domain_results), "domains": list(domain_results.keys()), "aggregated_metrics": {k: round(np.mean(v), 5) for k, v in aggregated.items()}}, f, indent=2)
            print(f"Saved: {config_dir}")

        # Print summary
        print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
        for (gen_name, setting), domain_results in all_results.items():
            avg = {m: np.mean([r["summary"]["overall_metrics"].get(m, 0) for r in domain_results.values()]) for m in ["rouge_l", "bertscore_f1", "meteor"]}
            print(f"  {gen_name} / {setting}: ROUGE-L={avg['rouge_l']:.4f}  BERT-F1={avg['bertscore_f1']:.4f}  METEOR={avg['meteor']:.4f}")
        print(f"\nResults saved to: {base_output_dir}")
        return {"output_dir": str(base_output_dir), "results": all_results}


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG Generation Pipeline for RECOR", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=str, help="Base directory with data folder")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--domains", type=str, help="Comma-separated domains")
    parser.add_argument("--generators", type=str, help="Comma-separated generators")
    parser.add_argument("--retrieval-settings", type=str, help="Comma-separated: retrieved,oracle,no_retrieval")
    parser.add_argument("--retrieval-cache", type=str, help="Path to pre-computed retrieval cache")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K documents (default: 10)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    config = Config()
    if args.base_dir:
        config.base_dir = args.base_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    config.use_gpu = not args.no_gpu
    config.gpu_device_id = args.gpu_id

    if config.use_gpu:
        gpu_info = detect_gpu()
        if gpu_info['available']:
            print(f"\n[OK] GPU: {gpu_info['device_name']} (CUDA {gpu_info['cuda_version']})")
        else:
            print("\n[WARN] GPU not available, using CPU")
            config.use_gpu = False

    parse_list = lambda s: [x.strip() for x in s.split(",")] if s else None
    settings = parse_list(args.retrieval_settings) or config.retrieval_settings

    if "retrieved" in settings and not args.retrieval_cache:
        print("ERROR: --retrieval-cache required for 'retrieved' setting")
        print("\nExamples:")
        print("  python rag_pipeline.py --retrieval-settings no_retrieval --generators vllm:model")
        print("  python rag_pipeline.py --retrieval-settings oracle --generators vllm:model")
        print("  python rag_pipeline.py --retrieval-cache path/to/cache --generators vllm:model")
        return

    pipeline = RAGPipeline(config)
    pipeline.run_generation(
        retrieval_cache=args.retrieval_cache,
        generators=parse_list(args.generators),
        retrieval_settings=parse_list(args.retrieval_settings),
        domains=parse_list(args.domains),
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
