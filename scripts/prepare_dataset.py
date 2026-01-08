"""
Dataset Preparation Script for RECOR Benchmark
===============================================

Copies dataset files from source directory to HuggingFace-ready structure.

Usage:
    python scripts/prepare_dataset.py --source <source_dir> --output <hf_dir>

Source files expected (11 domains):
    - {domain}_benchmark.jsonl
    - {domain}_positive_documents.jsonl
    - {domain}_negative_documents.jsonl

Domains: Drones, biology, earth_science, economics, hardware, law,
         medicalsciences, politics, psychology, robotics, sustainable_living
"""

import os
import shutil
import argparse
from pathlib import Path


def copy_dataset_files(source_dir: str, output_dir: str):
    """Copy dataset files to HuggingFace structure."""
    source = Path(source_dir)
    output = Path(output_dir)

    # Create output directories
    benchmark_dir = output / "data" / "benchmark"
    corpus_dir = output / "data" / "corpus"

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Copy benchmark files
    print("Copying benchmark files...")
    benchmark_count = 0
    for f in source.glob("*_benchmark.jsonl"):
        dest = benchmark_dir / f.name
        shutil.copy2(f, dest)
        print(f"  {f.name}")
        benchmark_count += 1
    print(f"  -> {benchmark_count} files copied")

    # Copy positive document files
    print("\nCopying positive document files...")
    pos_count = 0
    for f in source.glob("*_positive_documents.jsonl"):
        dest = corpus_dir / f.name
        shutil.copy2(f, dest)
        print(f"  {f.name}")
        pos_count += 1
    print(f"  -> {pos_count} files copied")

    # Copy negative document files
    print("\nCopying negative document files...")
    neg_count = 0
    for f in source.glob("*_negative_documents.jsonl"):
        dest = corpus_dir / f.name
        shutil.copy2(f, dest)
        print(f"  {f.name}")
        neg_count += 1
    print(f"  -> {neg_count} files copied")

    total = benchmark_count + pos_count + neg_count
    print(f"\nDone! {total} files copied to {output}")


def main():
    parser = argparse.ArgumentParser(description="Prepare RECOR dataset for HuggingFace")
    parser.add_argument("--source", "-s", required=True, help="Source directory with JSONL files")
    parser.add_argument("--output", "-o", required=True, help="Output HuggingFace directory")
    args = parser.parse_args()

    if not Path(args.source).exists():
        print(f"Error: Source directory not found: {args.source}")
        return

    copy_dataset_files(args.source, args.output)


if __name__ == "__main__":
    main()
