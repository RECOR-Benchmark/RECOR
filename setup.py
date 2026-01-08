"""
Setup script for RECOR benchmark.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="recor-benchmark",
    version="1.0.0",
    author="Anonymous",  # ACL Submission
    author_email="anonymous@example.com",
    description="RECOR: Reasoning-focused Multi-turn Conversational Retrieval Benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RECOR-Benchmark/RECOR",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "python-dotenv>=0.19.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "openai>=1.0.0",
        "rouge-score>=0.1.2",
        "bert-score>=0.3.12",
        "nltk>=3.8.0",
        "datasets>=2.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "all-embeddings": [
            "cohere>=4.0.0",
            "voyageai>=0.1.0",
            "google-generativeai>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "recor-retrieve=src.retrieval.retrievers:main",
            "recor-generate=src.generation.rag_pipeline:main",
            "recor-evaluate=src.evaluation.llm_judge:main",
        ],
    },
)
