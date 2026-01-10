import os
import json
import argparse
import time
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================
# CONFIGURATION - Set these paths for your environment
# =====================================================

# Base directory containing data folder
BASE_DIR = Path(os.getenv("RECOR_DATA_DIR", "."))

# Gen results directory
GEN_RESULTS_DIR = Path(os.getenv("RECOR_GEN_RESULTS", BASE_DIR / "gen_results"))

# Domain data folders (inside data/ subdirectory)
DATA_DIR = BASE_DIR / "data"
BENCHMARK_DIR = DATA_DIR / "benchmark"
CORPUS_DIR = DATA_DIR / "corpus"

# Domain classification
BRIGHT_DOMAINS = ["biology", "earth_science", "economics", "psychology", "robotics", "sustainable_living"]
ANNOTATED_DOMAINS = ["Drones", "hardware", "law", "medicalsciences", "politics"]

MODE_FOLDERS = {
    "no_retrieval": "Without any documents",
    "oracle": "Oracle [with gold docs]",
    "retrieved": "Retrieval k=5"
}

# =====================================================
# AZURE OPENAI CLIENT
# =====================================================

def get_azure_client():
    """Get Azure OpenAI client."""
    from openai import AzureOpenAI

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable not set")
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable not set")
    if not api_version:
        raise ValueError("AZURE_OPENAI_API_VERSION environment variable not set")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )
    return client


def get_deployment_name():
    """Get Azure deployment name from environment."""
    deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
    if not deployment_name:
        raise ValueError("AZURE_DEPLOYMENT_NAME environment variable not set")
    return deployment_name


# =====================================================
# DOCUMENT LOADING
# =====================================================

def load_documents_for_domain(domain: str) -> Dict[str, str]:
    """Load all documents for a domain."""
    documents = {}

    corpus_file = CORPUS_DIR / f"{domain}_documents.jsonl"

    # Load documents
    if corpus_file.exists():
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    documents[d["doc_id"]] = d["content"]
        logging.info(f"  Loaded {len(documents)} docs from {corpus_file.name}")
    else:
        logging.warning(f"  Corpus file not found: {corpus_file}")

    return documents


# =====================================================
# LLM JUDGE EVALUATOR
# =====================================================

class LLMJudgeEvaluator:
    def __init__(self, max_workers: int = 10, rate_limit_delay: float = 0.05):
        self.client = get_azure_client()
        self.model = get_deployment_name()
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.call_count = 0
        self.total_tokens = 0
        self.errors = 0
        self._lock = None

    def _build_prompt(
        self,
        question: str,
        reference: str,
        generated: str,
        sources: Optional[List[str]] = None
    ) -> str:
        """Build evaluation prompt - NO TRUNCATION."""

        if sources and len(sources) > 0:
            # With documents - evaluate faithfulness too (5 metrics)
            context = "\n\n---DOCUMENT---\n\n".join(sources)

            prompt = f"""Evaluate the Generated Answer against the Reference Answer.

[DOCUMENTS]
{context}

[QUESTION]
{question}

[REFERENCE ANSWER]
{reference}

[GENERATED ANSWER]
{generated}

Rate 1-10 for each criterion:

1. CORRECTNESS: Does Generated match Reference factually?
   1-3: Major factual errors or contradictions
   4-6: Some correct info but notable errors/omissions
   7-9: Mostly correct with minor issues
   10: Fully accurate

2. COMPLETENESS: Does Generated cover all key points from Reference?
   1-3: Missing most key information
   4-6: Covers some points, misses important ones
   7-9: Covers most points, minor omissions
   10: All key points covered

3. RELEVANCE: Does Generated directly answer the Question?
   1-3: Off-topic or doesn't address question
   4-6: Partially addresses question
   7-9: Addresses question with minor tangents
   10: Directly and fully answers question

4. COHERENCE: Is Generated well-organized and clear?
   1-3: Incoherent or confusing
   4-6: Understandable but poorly structured
   7-9: Clear with minor issues
   10: Excellent clarity and flow

5. FAITHFULNESS: Is Generated supported by the Documents?
   1-3: Contains claims not in documents (hallucination)
   4-6: Partially supported, some unsupported claims
   7-9: Mostly supported with minor extrapolations
   10: Fully grounded in documents

Output ONLY five integers separated by commas.
Example: 7,8,6,9,8"""
        else:
            # Without documents - no faithfulness (4 metrics)
            prompt = f"""Evaluate the Generated Answer against the Reference Answer.

[QUESTION]
{question}

[REFERENCE ANSWER]
{reference}

[GENERATED ANSWER]
{generated}

Rate 1-10 for each criterion:

1. CORRECTNESS: Does Generated match Reference factually?
   1-3: Major factual errors or contradictions
   4-6: Some correct info but notable errors/omissions
   7-9: Mostly correct with minor issues
   10: Fully accurate

2. COMPLETENESS: Does Generated cover all key points from Reference?
   1-3: Missing most key information
   4-6: Covers some points, misses important ones
   7-9: Covers most points, minor omissions
   10: All key points covered

3. RELEVANCE: Does Generated directly answer the Question?
   1-3: Off-topic or doesn't address question
   4-6: Partially addresses question
   7-9: Addresses question with minor tangents
   10: Directly and fully answers question

4. COHERENCE: Is Generated well-organized and clear?
   1-3: Incoherent or confusing
   4-6: Understandable but poorly structured
   7-9: Clear with minor issues
   10: Excellent clarity and flow

Output ONLY four integers separated by commas.
Example: 7,8,6,9"""

        return prompt

    def evaluate_single(
        self,
        question: str,
        reference: str,
        generated: str,
        sources: Optional[List[str]] = None,
        max_retries: int = 3
    ) -> Dict[str, float]:
        """Evaluate a single answer using LLM judge."""

        prompt = self._build_prompt(question, reference, generated, sources)
        has_sources = sources and len(sources) > 0

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=30,
                    temperature=0
                )

                self.call_count += 1
                if response.usage:
                    self.total_tokens += response.usage.total_tokens

                content = response.choices[0].message.content.strip()

                # Parse scores
                scores = []
                for part in content.replace(" ", "").split(","):
                    try:
                        scores.append(int(part.strip()))
                    except ValueError:
                        continue

                if has_sources:
                    # 5 metrics: correctness, completeness, relevance, coherence, faithfulness
                    if len(scores) >= 5:
                        return {
                            "llm_correctness": scores[0] / 10.0,
                            "llm_completeness": scores[1] / 10.0,
                            "llm_relevance": scores[2] / 10.0,
                            "llm_coherence": scores[3] / 10.0,
                            "llm_faithfulness": scores[4] / 10.0,
                            "llm_judge_avg": sum(scores[:5]) / 50.0
                        }
                    else:
                        raise ValueError(f"Expected 5 scores, got {len(scores)}: {content}")
                else:
                    # 4 metrics: correctness, completeness, relevance, coherence
                    if len(scores) >= 4:
                        return {
                            "llm_correctness": scores[0] / 10.0,
                            "llm_completeness": scores[1] / 10.0,
                            "llm_relevance": scores[2] / 10.0,
                            "llm_coherence": scores[3] / 10.0,
                            "llm_faithfulness": 0.0,  # N/A for no-retrieval
                            "llm_judge_avg": sum(scores[:4]) / 40.0
                        }
                    else:
                        raise ValueError(f"Expected 4 scores, got {len(scores)}: {content}")

            except Exception as e:
                logging.debug(f"Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))  # Linear backoff
                else:
                    self.errors += 1
                    logging.warning(f"All retries failed: {e}")
                    return {
                        "llm_correctness": 0.0,
                        "llm_completeness": 0.0,
                        "llm_relevance": 0.0,
                        "llm_coherence": 0.0,
                        "llm_faithfulness": 0.0,
                        "llm_judge_avg": 0.0,
                        "error": str(e)
                    }

    def evaluate_conversation_batch(
        self,
        turns: List[Dict[str, Any]]
    ) -> List[Dict[str, float]]:
        """
        Evaluate all turns in a conversation as a batch.
        Turns are processed in parallel within the conversation.
        """
        if not turns:
            return []

        results = [None] * len(turns)

        def eval_turn(idx: int, turn: Dict) -> Tuple[int, Dict]:
            time.sleep(self.rate_limit_delay * (idx % self.max_workers))
            result = self.evaluate_single(
                question=turn["question"],
                reference=turn["reference"],
                generated=turn["generated"],
                sources=turn.get("sources")
            )
            return idx, result

        with ThreadPoolExecutor(max_workers=min(len(turns), self.max_workers)) as executor:
            futures = {executor.submit(eval_turn, i, t): i for i, t in enumerate(turns)}

            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    def evaluate_domain_parallel(
        self,
        conversations: Dict[str, List[Dict[str, Any]]],
        progress_callback=None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate all conversations in a domain in parallel.

        Args:
            conversations: Dict mapping conv_id -> list of turns
            progress_callback: Optional callback for progress updates

        Returns:
            Dict mapping conv_id -> turn_id -> scores
        """
        results = {}
        total_turns = sum(len(turns) for turns in conversations.values())
        processed = 0

        def process_conversation(conv_id: str, turns: List[Dict]) -> Tuple[str, Dict]:
            conv_results = {}
            turn_scores = self.evaluate_conversation_batch(turns)

            for turn, scores in zip(turns, turn_scores):
                conv_results[turn["turn_key"]] = scores

            return conv_id, conv_results

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_conversation, conv_id, turns): conv_id
                for conv_id, turns in conversations.items()
            }

            pbar = tqdm(total=len(conversations), desc="Conversations")

            for future in as_completed(futures):
                conv_id, conv_results = future.result()
                results[conv_id] = conv_results
                processed += len(conv_results)
                pbar.update(1)

                if progress_callback:
                    progress_callback(processed, total_turns)

            pbar.close()

        return results


# =====================================================
# MAIN PROCESSING
# =====================================================

def group_by_conversation(answers: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    """Group turns by conversation ID."""
    conversations = defaultdict(list)

    for turn_key, data in answers.items():
        # turn_key format: "conv_id_turn_turn_id" e.g., "0_turn_1"
        parts = turn_key.rsplit("_turn_", 1)
        if len(parts) == 2:
            conv_id = parts[0]
        else:
            conv_id = turn_key

        conversations[conv_id].append({
            "turn_key": turn_key,
            "question": data["query"],
            "reference": data["reference_answer"],
            "generated": data["generated_answer"],
            "doc_ids": data.get("doc_ids", []),
            "sources": None  # Will be filled later
        })

    # Sort turns within each conversation
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: x["turn_key"])

    return dict(conversations)


def process_generator_folder(
    evaluator: LLMJudgeEvaluator,
    gen_folder: Path,
    mode: str,
    checkpoint_file: Path
) -> Dict[str, Any]:
    """Process all domains for a single generator folder."""

    # Load checkpoint if exists
    checkpoint = {}
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        logging.info(f"Loaded checkpoint with {len(checkpoint.get('results', {}))} domains")

    results = checkpoint.get("results", {})

    for domain_folder in sorted(gen_folder.iterdir()):
        if not domain_folder.is_dir():
            continue

        domain = domain_folder.name
        answers_file = domain_folder / "generated_answers.json"

        if not answers_file.exists():
            logging.warning(f"No generated_answers.json in {domain_folder}")
            continue

        # Skip if already processed
        if domain in results and len(results[domain]) > 0:
            logging.info(f"Skipping {domain} (already processed - {len(results[domain])} turns)")
            continue

        print(f"\n{'='*60}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")

        # Load documents for this domain
        documents = {}
        if mode != "no_retrieval":
            documents = load_documents_for_domain(domain)
            print(f"  Total documents loaded: {len(documents)}")

        # Load generated answers
        with open(answers_file, 'r', encoding='utf-8') as f:
            answers = json.load(f)
        print(f"  Total turns to evaluate: {len(answers)}")

        # Group by conversation
        conversations = group_by_conversation(answers)
        print(f"  Total conversations: {len(conversations)}")

        # Add document contents to turns
        if mode != "no_retrieval":
            missing_docs = 0
            for conv_id, turns in conversations.items():
                for turn in turns:
                    if turn["doc_ids"]:
                        sources = []
                        for doc_id in turn["doc_ids"]:
                            if doc_id in documents:
                                sources.append(documents[doc_id])
                            else:
                                missing_docs += 1
                        turn["sources"] = sources if sources else None
            if missing_docs > 0:
                print(f"  Warning: {missing_docs} document references not found")

        # Evaluate all conversations in parallel
        print(f"  Evaluating with {evaluator.max_workers} workers...")

        conv_results = evaluator.evaluate_domain_parallel(conversations)

        # Flatten results
        domain_results = {}
        for conv_id, conv_data in conv_results.items():
            domain_results.update(conv_data)

        results[domain] = domain_results

        # Save checkpoint after each domain
        checkpoint["results"] = results
        checkpoint["last_domain"] = domain
        checkpoint["api_calls"] = evaluator.call_count
        checkpoint["total_tokens"] = evaluator.total_tokens
        checkpoint["errors"] = evaluator.errors

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"  Completed: {len(domain_results)} turns evaluated")
        print(f"  API calls so far: {evaluator.call_count}, Tokens: {evaluator.total_tokens}, Errors: {evaluator.errors}")

    return results


def compute_summary(results: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, float]:
    """Compute aggregated summary statistics."""
    all_scores = {
        "llm_correctness": [],
        "llm_completeness": [],
        "llm_relevance": [],
        "llm_coherence": [],
        "llm_faithfulness": [],
        "llm_judge_avg": []
    }

    per_domain = {}

    for domain, domain_results in results.items():
        domain_scores = {k: [] for k in all_scores}

        for turn_key, scores in domain_results.items():
            if "error" not in scores:
                for metric in all_scores:
                    if metric in scores and scores[metric] > 0:
                        all_scores[metric].append(scores[metric])
                        domain_scores[metric].append(scores[metric])

        # Per-domain averages
        per_domain[domain] = {
            metric: round(sum(values) / len(values), 5) if values else 0.0
            for metric, values in domain_scores.items()
        }
        per_domain[domain]["count"] = len(domain_scores["llm_correctness"])

    # Overall averages
    summary = {
        metric: round(sum(values) / len(values), 5) if values else 0.0
        for metric, values in all_scores.items()
    }
    summary["total_evaluated"] = len(all_scores["llm_correctness"])
    summary["per_domain"] = per_domain

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-Judge evaluation on generated answers (Azure GPT-4o)")
    parser.add_argument("--mode", choices=["no_retrieval", "oracle", "retrieved", "all"], default="all",
                       help="Which mode to evaluate (default: all)")
    parser.add_argument("--generator", default=None, help="Specific generator to evaluate (default: all)")
    parser.add_argument("--domain", default=None, help="Specific domain to evaluate (default: all)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-workers", type=int, default=10, help="Max parallel workers (default: 10)")
    parser.add_argument("--rate-limit-delay", type=float, default=0.05, help="Delay between requests (default: 0.05)")

    args = parser.parse_args()

    evaluator = LLMJudgeEvaluator(
        max_workers=args.max_workers,
        rate_limit_delay=args.rate_limit_delay
    )

    modes_to_process = [args.mode] if args.mode != "all" else ["no_retrieval", "oracle", "retrieved"]

    print("=" * 80)
    print("LLM-as-Judge Evaluator (Azure GPT-4o)")
    print("=" * 80)
    print(f"Model: {evaluator.model}")
    print(f"Modes: {modes_to_process}")
    print(f"Max workers: {args.max_workers}")
    print(f"Resume: {args.resume}")
    print(f"Generator filter: {args.generator or 'all'}")
    print(f"Domain filter: {args.domain or 'all'}")
    print("=" * 80)

    all_summaries = {}

    for mode in modes_to_process:
        mode_folder = GEN_RESULTS_DIR / MODE_FOLDERS[mode]

        if not mode_folder.exists():
            logging.warning(f"Mode folder not found: {mode_folder}")
            continue

        print(f"\n{'#'*80}")
        print(f"# MODE: {mode.upper()}")
        print(f"{'#'*80}")

        for gen_folder in sorted(mode_folder.iterdir()):
            if not gen_folder.is_dir():
                continue

            gen_name = gen_folder.name

            # Filter by generator if specified
            if args.generator and args.generator.lower() not in gen_name.lower():
                continue

            print(f"\n{'='*60}")
            print(f"Generator: {gen_name}")
            print(f"{'='*60}")

            # Checkpoint and output files
            checkpoint_file = gen_folder / "llm_judge_checkpoint.json"
            output_file = gen_folder / "llm_judge_results.json"
            summary_file = gen_folder / "llm_judge_summary.json"

            # Skip if already complete and not resuming
            if output_file.exists() and not args.resume:
                print(f"  Already complete. Use --resume to re-evaluate.")
                # Load existing summary
                if summary_file.exists():
                    with open(summary_file, 'r') as f:
                        all_summaries[(mode, gen_name)] = json.load(f)
                continue

            # Process
            results = process_generator_folder(
                evaluator=evaluator,
                gen_folder=gen_folder,
                mode=mode,
                checkpoint_file=checkpoint_file
            )

            # Save final results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            # Compute and save summary
            summary = compute_summary(results)
            summary["generator"] = gen_name
            summary["mode"] = mode
            summary["api_calls"] = evaluator.call_count
            summary["total_tokens"] = evaluator.total_tokens
            summary["errors"] = evaluator.errors

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)

            all_summaries[(mode, gen_name)] = summary

            print(f"\nResults saved to: {output_file}")
            print(f"Summary: avg={summary.get('llm_judge_avg', 0):.3f}, "
                  f"correct={summary.get('llm_correctness', 0):.3f}, "
                  f"complete={summary.get('llm_completeness', 0):.3f}, "
                  f"relevant={summary.get('llm_relevance', 0):.3f}, "
                  f"coherent={summary.get('llm_coherence', 0):.3f}, "
                  f"faithful={summary.get('llm_faithfulness', 0):.3f}")

            # Clean up checkpoint
            if checkpoint_file.exists():
                checkpoint_file.unlink()

    # Print final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    if all_summaries:
        print(f"\n{'Generator':<50} {'Mode':<12} {'Corr':<6} {'Comp':<6} {'Relv':<6} {'Cohr':<6} {'Faith':<6} {'Avg':<6}")
        print("-" * 100)

        for (mode, gen_name), summary in sorted(all_summaries.items(), key=lambda x: (x[0][0], -x[1].get('llm_judge_avg', 0))):
            short_name = gen_name[:47] + "..." if len(gen_name) > 50 else gen_name
            print(f"{short_name:<50} {mode:<12} "
                  f"{summary.get('llm_correctness', 0):.2f}  "
                  f"{summary.get('llm_completeness', 0):.2f}  "
                  f"{summary.get('llm_relevance', 0):.2f}  "
                  f"{summary.get('llm_coherence', 0):.2f}  "
                  f"{summary.get('llm_faithfulness', 0):.2f}  "
                  f"{summary.get('llm_judge_avg', 0):.2f}")

    print("\n" + "=" * 80)
    print(f"Total API calls: {evaluator.call_count}")
    print(f"Total tokens: {evaluator.total_tokens}")
    print(f"Total errors: {evaluator.errors}")
    estimated_cost = (evaluator.total_tokens / 1000) * 0.005  # GPT-4o rough estimate
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
