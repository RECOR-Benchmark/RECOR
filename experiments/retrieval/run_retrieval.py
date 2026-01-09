import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from .retrievers import RETRIEVAL_FUNCS, calculate_retrieval_metrics

def load_corpus(corpus_dir, domain):
    """Load all documents from a domain's corpus directory"""
    domain_dir = Path(corpus_dir) / domain
    doc_ids = []
    documents = []

    # Get all .txt files
    txt_files = sorted(domain_dir.glob("*.txt"))

    for txt_file in tqdm(txt_files, desc=f"Loading {domain} corpus"):
        # Document ID is relative path: domain/filename.txt
        doc_id = f"{domain}/{txt_file.name}"
        doc_ids.append(doc_id)

        # Read document content
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        documents.append(content)

    return doc_ids, documents

def load_queries(queries_dir, domain):
    """Load queries from domain's JSONL file"""
    query_file = Path(queries_dir) /f"{domain}_queries.jsonl"

    queries = []
    query_ids = []
    gold_ids_map = {}
    excluded_ids = {}

    if not query_file.exists():
        print(f"Warning: Query file not found: {query_file}")
        return queries, query_ids, gold_ids_map, excluded_ids

    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if len(data['gold_ids']) > 0:
                query_id = data['id']
                query_ids.append(query_id)
                queries.append(data['query'])

                # Store gold IDs for evaluation
                gold_ids_map[query_id] = data['gold_ids']

                # Store negative IDs as excluded
                excluded_ids[query_id] = data.get('negative_ids', [])



    return queries, query_ids, gold_ids_map, excluded_ids

def evaluate_domain(domain, dataset_dir, args, config):
    """Evaluate a single domain"""
    print("\n" + "="*80)
    print(f"EVALUATING DOMAIN: {domain.upper()}")
    print("="*80)

    corpus_dir = Path(dataset_dir) / "corpus"
    queries_dir = Path(dataset_dir) / args.model_data_dir

    # Load corpus for this domain only
    print(f"\nLoading {domain} corpus...")
    doc_ids, documents = load_corpus(corpus_dir, domain)
    print(f"  Documents: {len(documents)}")

    if len(documents) == 0:
        print(f"No documents found for {domain}, skipping...")
        return None

    # Load queries for this domain only
    print(f"\nLoading {domain} queries...")
    queries, query_ids, gold_ids_map, excluded_ids = load_queries(queries_dir, domain)
    print(f"  Queries: {len(queries)}")

    if len(queries) == 0:
        print(f"No queries found for {domain}, skipping...")
        return None

    # Verify no overlap between gold and excluded
    for qid in query_ids:
        overlap = set(excluded_ids.get(qid, [])).intersection(set(gold_ids_map.get(qid, [])))
        assert len(overlap) == 0, f"Query {qid} has overlap between gold and excluded: {overlap}"

    if args.debug:
        print("\n[DEBUG MODE] Using only first 30 documents")
        documents = documents[:30]
        doc_ids = doc_ids[:30]

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

    score_file_path = os.path.join(domain_output_dir, 'scores.json')

    # Run retrieval
    if not os.path.isfile(score_file_path):
        print(f"\nRunning {args.model} retrieval on {domain}...")

        scores = RETRIEVAL_FUNCS[args.model](
            queries=queries,
            query_ids=query_ids,
            documents=documents,
            doc_ids=doc_ids,
            task=domain,  # Use domain as task name for caching
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

    # Prepare ground truth in pytrec_eval format
    ground_truth = {}
    for qid in query_ids:
        ground_truth[qid] = {}
        for gold_id in gold_ids_map.get(qid, []):
            ground_truth[qid][gold_id] = 1

    # Verify excluded IDs are not in scores or ground truth
    for qid in query_ids:
        for excluded_id in excluded_ids.get(qid, []):
            assert excluded_id not in scores.get(qid, {}), f"Excluded {excluded_id} in scores for {qid}"
            assert excluded_id not in ground_truth.get(qid, {}), f"Excluded {excluded_id} in ground truth for {qid}"

    # Calculate metrics
    print(f"\n{domain.upper()} RESULTS:")
    results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)

    with open(os.path.join(domain_output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return {
        'domain': domain,
        'num_queries': len(queries),
        'num_documents': len(documents),
        'results': results
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to dataset directory (contains corpus/, queries/, images/)')
    parser.add_argument('--domains', type=str, nargs='+',
                        default=['biology', 'Drones', 'earth_science', 'economics',
                                'hardware', 'law', 'medicalsciences', 'politics',
                                'psychology', 'robotics', 'sustainable_living'],
                        help='List of domains to evaluate')
    parser.add_argument('--model_data_dir', type=str, required=True)

    parser.add_argument('--model', type=str, required=True,
                        choices=['bm25','cohere','e5','google','grit','inst-l','inst-xl',
                                 'openai','qwen','qwen2','sbert','sf','voyage','bge',
                                 'nomic', 'm2', 'contriever', 'reasonir', 'rader', 'diver-retriever'])
    parser.add_argument('--query_max_length', type=int, default=-1)
    parser.add_argument('--doc_max_length', type=int, default=-1)
    parser.add_argument('--encode_batch_size', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--config_dir', type=str, default='configs')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--key', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ignore_cache', action='store_true')
    args = parser.parse_args()

    # Create main output directory
    args.output_dir = os.path.join(args.output_dir, f"multimodal_ir_{args.model}")
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
        result = evaluate_domain(domain, args.dataset_dir, args, config)

        if result is not None:
            all_domain_results[domain] = result

    # Save aggregated results
    print("\n" + "="*80)
    print("AGGREGATED RESULTS")
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

        # Save summary
        summary = {
            'model': args.model,
            'num_domains': len(all_domain_results),
            'domains': list(all_domain_results.keys()),
            'aggregated_metrics': aggregated_metrics,
            'per_domain': all_domain_results
        }

        with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n Evaluation complete! Results saved to {args.output_dir}")
        print(f"  - Per-domain results: {args.output_dir}/<domain>/results.json")
        print(f"  - Aggregated summary: {args.output_dir}/summary.json")
    else:
        print("\nNo domains were successfully evaluated.")


if __name__ == '__main__':
    main()
