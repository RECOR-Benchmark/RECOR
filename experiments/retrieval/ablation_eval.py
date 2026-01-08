"""
Ablation Study for Query Augmentation Strategies

This script evaluates how different query augmentation strategies affect retrieval:
  - --append-history: Add conversation history to query
  - --append-reasoning: Add subquestion reasoning to query
  - --append-reasoning-metadata: Add target info and relevance signals

Paper Reference: Section 5.1 (Ablation Studies)
"""

import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from retrievers import RETRIEVAL_FUNCS, calculate_retrieval_metrics

def load_corpus_from_json(queries_dir, domain):
    """Load all documents from positive and negative JSON files"""
    positive_file = Path(queries_dir) / f"{domain}_positive_documents.jsonl"
    negative_file = Path(queries_dir) / f"{domain}_negative_documents.jsonl"

    doc_ids = []
    documents = []

    # Load positive documents
    if positive_file.exists():
        print(f"  Loading positive documents from {positive_file}")
        with open(positive_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                doc_ids.append(data['doc_id'])
                documents.append(data['content'])

    # Load negative documents
    if negative_file.exists():
        print(f"  Loading negative documents from {negative_file}")
        with open(negative_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                doc_ids.append(data['doc_id'])
                documents.append(data['content'])

    return doc_ids, documents

def load_conversational_queries(queries_dir, domain):
    """Load conversational queries from benchmark JSONL file"""
    query_file = Path(queries_dir) / f"{domain}_benchmark.jsonl"

    conversations = []

    if not query_file.exists():
        print(f"Warning: Query file not found: {query_file}")
        return conversations

    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            conversations.append(data)

    return conversations

def evaluate_domain_conversational(domain, dataset_dir, args, config):
    """Evaluate a single domain with conversational queries"""
    print("\n" + "="*80)
    print(f"EVALUATING DOMAIN: {domain.upper()}")
    print("="*80)

    queries_dir = Path(dataset_dir) / args.model_data_dir

    # Load corpus from JSON files
    print(f"\nLoading {domain} corpus from JSON files...")
    doc_ids, documents = load_corpus_from_json(queries_dir, domain)
    print(f"  Total documents: {len(documents)}")

    if len(documents) == 0:
        print(f"No documents found for {domain}, skipping...")
        return None

    # Load conversational queries
    print(f"\nLoading {domain} conversational queries...")
    conversations = load_conversational_queries(queries_dir, domain)
    print(f"  Conversations: {len(conversations)}")

    if len(conversations) == 0:
        print(f"No conversations found for {domain}, skipping...")
        return None

    # Debug: Show structure of first conversation
    if conversations:
        first_conv = conversations[0]
        print(f"  First conversation ID: {first_conv.get('id', 'N/A')}")
        if 'turns' in first_conv and first_conv['turns']:
            first_turn = first_conv['turns'][0]
            print(f"  First turn keys: {list(first_turn.keys())}")
            # Identify which gold field is present
            if 'gold_doc_ids' in first_turn:
                print(f"  Using field: 'gold_doc_ids'")
            elif 'gold_ids' in first_turn:
                print(f"  Using field: 'gold_ids'")
            else:
                print(f"  WARNING: Neither 'gold_doc_ids' nor 'gold_ids' found!")

    # Prepare all queries from all turns (so model is loaded only once)
    all_queries = []
    all_query_ids = []
    gold_ids_map = {}
    excluded_ids = {}
    query_to_conversation = {}  # Map query_id to (conversation_id, turn_id)

    for conv in conversations:
        conv_id = conv['id']
        for turn in conv['turns']:
            turn_id = turn['turn_id']
            query_id = f"{conv_id}_turn_{turn_id}"

            # Handle both possible field names for gold documents
            if 'gold_doc_ids' in turn:
                gold_ids = turn['gold_doc_ids']
            elif 'gold_ids' in turn:
                gold_ids = turn['gold_ids']
            else:
                gold_ids = []

            # Skip queries with no gold documents
            if len(gold_ids) == 0:
                print(f"  Skipping query {query_id} - no gold documents")
                continue

            # Start with base query
            query_parts = [turn['query']]

            # Append conversation history if enabled
            if args.append_history:
                history = turn.get('conversation_history', '')
                if history and history != "No previous conversation.":
                    query_parts.append(f"Conversation History:\n{history}")

            # Append subquestion reasoning if enabled
            if args.append_reasoning:
                reasoning = turn.get('subquestion_reasoning', '')
                if reasoning:
                    query_parts.append(f"Reasoning:\n{reasoning}")

            # Append subquestion reasoning metadata if enabled
            if args.append_reasoning_metadata:
                metadata = turn.get('subquestion_reasoning_metadata', {})
                if metadata:
                    metadata_text = []

                    if 'target_information' in metadata:
                        metadata_text.append(f"Target Information: {metadata['target_information']}")

                    if 'relevance_signals' in metadata and metadata['relevance_signals']:
                        signals = ", ".join(metadata['relevance_signals'])
                        metadata_text.append(f"Relevance Signals: {signals}")

                    if 'irrelevance_signals' in metadata and metadata['irrelevance_signals']:
                        signals = ", ".join(metadata['irrelevance_signals'])
                        metadata_text.append(f"Irrelevance Signals: {signals}")

                    if metadata_text:
                        query_parts.append(f"Reasoning Metadata:\n" + "\n".join(metadata_text))

            # Combine all parts: query + history (if enabled) + reasoning (if enabled) + metadata (if enabled)
            query = "\n\n".join(query_parts)

            all_queries.append(query)
            all_query_ids.append(query_id)
            gold_ids_map[query_id] = gold_ids
            excluded_ids[query_id] = []  # No excluded IDs in this format
            query_to_conversation[query_id] = (conv_id, turn_id)

    print(f"  Total queries (all turns): {len(all_queries)}")

    if len(all_queries) == 0:
        print(f"\n  Warning: No valid queries found for {domain} (all queries have no gold documents)")
        return None

    # Show experimental conditions
    if args.append_history or args.append_reasoning or args.append_reasoning_metadata:
        print(f"\n  Experimental Conditions:")
        if args.append_history:
            print(f"    Appending conversation history to queries")
        if args.append_reasoning:
            print(f"    Appending subquestion reasoning to queries")
        if args.append_reasoning_metadata:
            print(f"    Appending reasoning metadata (target info, signals) to queries")

        if args.append_history and args.append_reasoning and args.append_reasoning_metadata:
            print(f"\n  Query Structure: Query + History + Reasoning + Metadata")
        elif args.append_history and args.append_reasoning:
            print(f"\n  Query Structure: Query + History + Reasoning")
        elif args.append_history and args.append_reasoning_metadata:
            print(f"\n  Query Structure: Query + History + Metadata")
        elif args.append_reasoning and args.append_reasoning_metadata:
            print(f"\n  Query Structure: Query + Reasoning + Metadata")
        elif args.append_history:
            print(f"\n  Query Structure: Query + History")
        elif args.append_reasoning:
            print(f"\n  Query Structure: Query + Reasoning")
        elif args.append_reasoning_metadata:
            print(f"\n  Query Structure: Query + Metadata")

        # Show example of modified query
        if all_queries:
            print(f"\n  Example modified query (first turn):")
            example_query = all_queries[0]
            preview = example_query[:300] + "..." if len(example_query) > 300 else example_query
            print(f"    {preview}")

    if args.debug:
        print("\n[DEBUG MODE] Using only first 30 documents and 5 queries")
        documents = documents[:30]
        doc_ids = doc_ids[:30]
        all_queries = all_queries[:5]
        all_query_ids = all_query_ids[:5]

    # Prepare kwargs
    kwargs = {}
    if args.query_max_length > 0:
        kwargs['query_max_length'] = args.query_max_length
    if args.doc_max_length > 0:
        kwargs['doc_max_length'] = args.doc_max_length
    if args.encode_batch_size > 0:
        kwargs['batch_size'] = args.encode_batch_size
    if args.key is not None:
        kwargs['key'] = args.key
    if args.ignore_cache:
        kwargs['ignore_cache'] = args.ignore_cache
    if args.checkpoint:
        kwargs['checkpoint'] = args.checkpoint

    # Domain-specific output directory
    domain_output_dir = os.path.join(args.output_dir, domain)
    os.makedirs(domain_output_dir, exist_ok=True)

    score_file_path = os.path.join(domain_output_dir, 'all_scores.json')

    # Run retrieval for ALL queries at once (model loaded only once!)
    if not os.path.isfile(score_file_path):
        print(f"\nRunning {args.model} retrieval on ALL {len(all_queries)} queries...")
        print("(Model will be loaded only once for all queries)")

        scores = RETRIEVAL_FUNCS[args.model](
            queries=all_queries,
            query_ids=all_query_ids,
            documents=documents,
            doc_ids=doc_ids,
            task=domain,
            instructions=config['instructions'],
            excluded_ids=excluded_ids,
            cache_dir=args.cache_dir,
            long_context=False,
            model_id=args.model,
            **kwargs
        )

        with open(score_file_path, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f"Scores saved to {score_file_path}")
    else:
        print(f"Loading existing scores from {score_file_path}")
        with open(score_file_path) as f:
            scores = json.load(f)

    # Evaluate per turn and aggregate
    print(f"\nEvaluating {len(conversations)} conversations...")
    conversation_results = {}

    for conv in tqdm(conversations, desc="Processing conversations"):
        conv_id = conv['id']
        conversation_results[conv_id] = {
            'turns': {},
            'num_turns': len(conv['turns']),
            'task': conv.get('task', domain)
        }

        for turn in conv['turns']:
            turn_id = turn['turn_id']
            query_id = f"{conv_id}_turn_{turn_id}"

            # Skip if query not in scores (debug mode)
            if query_id not in scores:
                continue

            # Get gold doc IDs - handle both possible field names
            if 'gold_doc_ids' in turn:
                gold_doc_ids = turn['gold_doc_ids']
            elif 'gold_ids' in turn:
                gold_doc_ids = turn['gold_ids']
            else:
                gold_doc_ids = []

            # Skip if no gold documents
            if len(gold_doc_ids) == 0:
                print(f"Warning: Skipping query {query_id} - no gold documents")
                continue

            # Prepare ground truth for this turn
            ground_truth = {
                query_id: {gold_id: 1 for gold_id in gold_doc_ids}
            }

            # Get scores for this turn
            turn_scores = {query_id: scores[query_id]}

            # Calculate metrics for this turn
            try:
                turn_results = calculate_retrieval_metrics(
                    results=turn_scores,
                    qrels=ground_truth
                )
            except Exception as e:
                print(f"Warning: Error calculating metrics for query {query_id}: {e}")
                continue

            conversation_results[conv_id]['turns'][turn_id] = {
                'query': turn['query'],
                'metrics': turn_results,
                'num_gold_docs': len(gold_doc_ids)
            }

    # Calculate average across turns for each conversation
    print("\nCalculating per-conversation averages...")
    for conv_id in list(conversation_results.keys()):
        turn_results_list = [
            tr['metrics'] for tr in conversation_results[conv_id]['turns'].values()
        ]

        # Skip conversations with no valid turns
        if not turn_results_list:
            print(f"Warning: Conversation {conv_id} has no valid turns, removing from results")
            del conversation_results[conv_id]
            continue

        avg_metrics = {}
        if turn_results_list:
            metric_keys = turn_results_list[0].keys()
            for key in metric_keys:
                values = [tr[key] for tr in turn_results_list]
                avg_metrics[key] = round(sum(values) / len(values), 5)

        conversation_results[conv_id]['average_across_turns'] = avg_metrics

    # Calculate overall average across all conversations
    print("Calculating overall dataset averages...")
    all_conv_averages = [
        cr['average_across_turns']
        for cr in conversation_results.values()
        if 'average_across_turns' in cr
    ]

    overall_metrics = {}
    if all_conv_averages:
        metric_keys = all_conv_averages[0].keys()
        for key in metric_keys:
            values = [ca[key] for ca in all_conv_averages]
            overall_metrics[key] = round(sum(values) / len(values), 5)
    else:
        print(f"\n  Warning: No valid conversations found for {domain}")
        return None

    print(f"\n{domain.upper()} OVERALL RESULTS (averaged across conversations):")
    print(json.dumps(overall_metrics, indent=2))

    # Show filtering statistics
    valid_convs = len(conversation_results)
    total_convs = len(conversations)
    if valid_convs < total_convs:
        print(f"\nNote: {total_convs - valid_convs} conversation(s) skipped (no valid turns with gold documents)")

    # Save detailed results
    detailed_results = {
        'overall_metrics': overall_metrics,
        'num_conversations': len(conversation_results),  # Valid conversations
        'total_conversations': len(conversations),  # Original count
        'total_turns': len([t for c in conversation_results.values() for t in c['turns'].values()]),  # Valid turns
        'conversation_results': conversation_results
    }

    with open(os.path.join(domain_output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2)

    # Save summary metrics
    with open(os.path.join(domain_output_dir, 'summary.json'), 'w') as f:
        json.dump({
            'domain': domain,
            'num_conversations': len(conversation_results),
            'total_conversations': len(conversations),
            'total_turns': len([t for c in conversation_results.values() for t in c['turns'].values()]),
            'overall_metrics': overall_metrics
        }, f, indent=2)

    return {
        'domain': domain,
        'num_conversations': len(conversation_results),  # Valid conversations only
        'total_conversations': len(conversations),  # Original count
        'total_turns': len([t for c in conversation_results.values() for t in c['turns'].values()]),  # Valid turns only
        'results': overall_metrics,
        'detailed_results': detailed_results
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to dataset directory (contains data/ folder with benchmark files)')
    parser.add_argument('--domains', type=str, nargs='+',
                        default=['biology', 'Drones', 'earth_science', 'economics',
                                'hardware', 'law', 'medicalsciences', 'politics',
                                'psychology', 'robotics', 'sustainable_living'],
                        help='List of domains to evaluate')
    parser.add_argument('--model_data_dir', type=str, default='data',
                        help='Subdirectory containing the benchmark JSONL files')

    parser.add_argument('--model', type=str, required=True,
                        choices=['bm25','cohere','e5','google','grit','inst-l','inst-xl',
                                 'openai','qwen','qwen2','sbert','sf','voyage','bge',
                                 'bge_ce', 'nomic', 'm2', 'contriever', 'reasonir', 'rader', 'diver-retriever'])
    parser.add_argument('--query_max_length', type=int, default=-1)
    parser.add_argument('--doc_max_length', type=int, default=-1)
    parser.add_argument('--encode_batch_size', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='outputs_conversational')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--config_dir', type=str, default='configs')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ignore_cache', action='store_true')

    # Experimental conditions for conversational IR
    parser.add_argument('--append-history', action='store_true',
                        help='Append conversation history to each query')
    parser.add_argument('--append-reasoning', action='store_true',
                        help='Append subquestion reasoning to each query')
    parser.add_argument('--append-reasoning-metadata', action='store_true',
                        help='Append subquestion reasoning metadata (target_information, relevance_signals, irrelevance_signals) to each query')

    args = parser.parse_args()

    # Create main output directory with experimental conditions
    output_suffix = f"conversational_ir_{args.model}"
    if args.append_history:
        output_suffix += "_with_history"
    if args.append_reasoning:
        output_suffix += "_with_reasoning"
    if args.append_reasoning_metadata:
        output_suffix += "_with_metadata"

    args.output_dir = os.path.join(args.output_dir, output_suffix)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model config
    config = {
        'instructions': {
            'query': 'Represent this query for retrieving relevant documents: ',
            'document': ''
        }
    }

    # If you have a config file:
    config_file = os.path.join(args.config_dir, args.model, "default.json")
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)

    # Evaluate each domain separately
    all_domain_results = {}

    for domain in args.domains:
        result = evaluate_domain_conversational(domain, args.dataset_dir, args, config)

        if result is not None:
            all_domain_results[domain] = result

    # Save aggregated results
    print("\n" + "="*80)
    print("AGGREGATED RESULTS ACROSS ALL DOMAINS")
    print("="*80)

    # Calculate average metrics across all domains
    if len(all_domain_results) > 0:
        # Get all metric keys from first domain
        first_domain = list(all_domain_results.values())[0]
        metric_keys = list(first_domain['results'].keys())

        aggregated_metrics = {}
        for key in metric_keys:
            values = [result['results'][key] for result in all_domain_results.values()]
            aggregated_metrics[key] = round(sum(values) / len(values), 5)

        print("\nAverage across all domains:")
        print(json.dumps(aggregated_metrics, indent=2))

        # Calculate total conversations and turns
        total_conversations = sum(r['num_conversations'] for r in all_domain_results.values())
        total_turns = sum(r['total_turns'] for r in all_domain_results.values())

        # Save summary
        summary = {
            'model': args.model,
            'experimental_conditions': {
                'append_history': args.append_history,
                'append_reasoning': args.append_reasoning,
                'append_reasoning_metadata': args.append_reasoning_metadata
            },
            'num_domains': len(all_domain_results),
            'total_conversations': total_conversations,  # Valid conversations
            'total_turns': total_turns,  # Valid turns
            'domains': list(all_domain_results.keys()),
            'aggregated_metrics': aggregated_metrics,
            'per_domain_summary': {
                domain: {
                    'num_conversations': r['num_conversations'],
                    'total_conversations': r['total_conversations'],
                    'total_turns': r['total_turns'],
                    'metrics': r['results']
                }
                for domain, r in all_domain_results.items()
            }
        }

        with open(os.path.join(args.output_dir, 'final_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n Evaluation complete! Results saved to {args.output_dir}")
        print(f"  - Per-domain detailed results: {args.output_dir}/<domain>/detailed_results.json")
        print(f"  - Per-domain summary: {args.output_dir}/<domain>/summary.json")
        print(f"  - Final aggregated summary: {args.output_dir}/final_summary.json")
        print(f"\n  Total conversations evaluated: {total_conversations}")
        print(f"  Total turns evaluated: {total_turns}")
    else:
        print("\nNo domains were successfully evaluated.")
