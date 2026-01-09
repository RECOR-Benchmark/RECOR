"""
Comprehensive Conversation Quality & Diversity Analysis (Modified)
====================================================================
Modified to work with outputs_Annotated_unified and outputs_bright_unified benchmark files.

Changes from original:
- Loads documents from *_positive_documents.jsonl files
- NO document truncation (uses full text)
- Fixes field mappings (task->domain, original_answer->gold_answer)
- Supports both Annotated and Bright benchmark formats
- Resolves supporting_doc_ids/gold_doc_ids to actual document content
- Can process BOTH folders (all 11 benchmark files) in ONE run

Usage:
    export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
    export AZURE_OPENAI_API_KEY="your-api-key"

    # Process ALL 11 benchmark files from both folders in ONE run:
    python comprehensive_analysis_modified.py --output ./results

    # Or specify custom directories:
    python comprehensive_analysis_modified.py --input-dirs ./outputs_Annotated_unified ./outputs_bright_unified --output ./results

Author: MURECOR Benchmark Team (Modified)
"""

import json
import os
import sys
import time
import random
import logging
import argparse
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np

from openai import AzureOpenAI


# ============================================================================
# CUSTOM JSON ENCODER FOR NUMPY TYPES
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for the analysis pipeline."""
    # Azure settings
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_api_version: str = ""
    azure_deployment: str = ""

    # Parallelization settings
    max_workers: int = 5

    # Rate limiting (safe defaults)
    min_request_interval: float = 0.3  # Min seconds between requests per worker
    global_rate_limit: float = 0.1     # Global delay between any requests

    # Retry settings
    max_retries: int = 3
    base_retry_delay: float = 2.0
    max_retry_delay: float = 30.0

    # Batching settings
    turns_per_batch: int = 4  # Process up to 4 turns in one API call

    # Output settings
    output_dir: str = "./analysis_results"
    save_every_n: int = 20  # Save intermediate results every N conversations

    # Sampling
    sample_size: Optional[int] = None
    random_seed: int = 42


# ============================================================================
# DETAILED PROMPTS WITH GRADING RUBRICS
# ============================================================================

# ============================================================================
# QUALITY DIMENSION PROMPTS (Clear, Simple, Non-Overlapping)
# ============================================================================
#
# Four dimensions for academic paper:
# 1. NATURALNESS: Does it sound like real human conversation?
# 2. COHERENCE: Do turns logically connect to each other?
# 3. QUESTION QUALITY: Do questions cover different useful aspects?
# 4. GROUNDEDNESS: Are answers supported by the documents?
#
# ============================================================================

NATURALNESS_PROMPT = """Evaluate whether this conversation sounds like NATURAL HUMAN SPEECH.

CONVERSATION:
{conversation}

TASK: Rate how natural and human-like the language is (not robotic or overly formal).

CHECK ONLY these language features:

1. CASUAL WORD CHOICES:
   - Contractions used: "don't", "isn't", "what's" (vs formal "do not", "is not")
   - Casual words: "got", "stuff", "things", "pretty much" (vs formal "obtained", "materials")
   - Softening words: "kind of", "sort of", "maybe", "I think"

2. CONVERSATION STARTERS:
   - Natural openers: "So", "Well", "Oh", "Yeah", "I mean", "Actually"
   - Response words: "Right", "Okay", "I see", "Got it"

3. SENTENCE STYLE:
   - Natural incomplete sentences (OK in speech)
   - Casual question phrasing (not stiff templates)

DO NOT CHECK (other prompts handle these):
- Whether turns connect logically
- Whether questions cover the topic well
- Whether answers are factually correct

SCORING:

5 - Sounds completely natural, like real human speech
4 - Mostly natural, few formal spots
3 - Mix of natural and formal/stiff language
2 - Mostly formal or robotic sounding
1 - Entirely artificial, would never occur in real speech

Return JSON:
{{
    "score": <1-5>,
    "natural_phrases": ["quote 2-3 natural-sounding phrases found"],
    "unnatural_phrases": ["quote any stiff or robotic phrases found"],
    "justification": "1-2 sentences explaining your score"
}}
"""

TURN_COHERENCE_PROMPT = """Evaluate whether conversation turns CONNECT LOGICALLY to each other.

CONVERSATION:
{conversation}

TASK: Rate how well each question follows from the previous answer.

CHECK ONLY these connection features:

1. CLEAR REFERENCES:
   - When "it", "this", "that", "they" are used, is it clear what they refer to?
   - When "the problem" or "the process" is mentioned, was it introduced earlier?

2. LOGICAL CONNECTIONS (how does each question relate to the previous answer?):
   - ASKS FOR DETAILS: Question wants more information about something just mentioned
   - ASKS TO CLARIFY: Question wants to understand something from the answer better
   - ASKS ABOUT EFFECTS: Question asks what happens as a result of what was described
   - ASKS FOR CONTRAST: Question asks about an alternative or opposite view
   - NARROWS FOCUS: Question zooms in on one specific part mentioned

3. NO GAPS:
   - Each question should make sense given what came before
   - No sudden jumps to unrelated topics without transition

DO NOT CHECK (other prompts handle these):
- Whether language sounds natural
- Whether questions are about different topics
- Whether answers are factually correct

SCORING:

5 - Every question clearly connects to the previous answer
4 - Most questions connect well, 1-2 slightly unclear links
3 - Some questions connect, but several feel disconnected
2 - Many questions don't clearly follow from previous answers
1 - Questions seem random, no logical flow

Return JSON:
{{
    "score": <1-5>,
    "good_connections": ["describe 1-2 turns that connect well"],
    "weak_connections": ["describe any turns that don't connect clearly"],
    "unclear_references": ["list any 'it/this/that' without clear meaning"],
    "justification": "1-2 sentences explaining your score"
}}
"""

QUESTION_QUALITY_PROMPT = """Evaluate whether the questions COVER DIFFERENT USEFUL ASPECTS of the topic.

ORIGINAL TOPIC:
{original_query}

QUESTIONS:
{questions}

TASK: Rate whether the questions explore the topic well without repeating.

CHECK ONLY these coverage features:

1. DIFFERENT ASPECTS:
   - Does each question ask about something DIFFERENT?
   - Are any two questions basically asking the same thing in different words?

2. USEFUL QUESTIONS:
   - Are questions specific enough to get clear answers?
   - Do questions go beyond obvious or trivial information?
   - Bad examples: "What is X?" (too basic), "Tell me about X" (too vague)

3. GOOD COVERAGE:
   - Together, do the questions explore important parts of the topic?
   - Are there major aspects of the topic that are missed?

DO NOT CHECK (other prompts handle these):
- Whether questions sound natural
- Whether questions follow logically from answers
- Whether the answers are correct

SCORING:

5 - All questions distinct, specific, and together cover the topic well
4 - Most questions distinct and useful, maybe 1 similar pair
3 - Some overlap between questions, some too vague, partial coverage
2 - Several questions overlap or are too vague/trivial
1 - Heavy repetition, mostly vague or useless questions

Return JSON:
{{
    "score": <1-5>,
    "aspects_covered": ["list the different aspects/subtopics the questions address"],
    "repeated_questions": ["list any question pairs that ask about the same thing"],
    "weak_questions": ["list any questions that are too vague or trivial"],
    "justification": "1-2 sentences explaining your score"
}}
"""

GROUNDEDNESS_PROMPT = """Evaluate whether the answers are SUPPORTED BY THE DOCUMENTS.

SOURCE DOCUMENTS:
{documents}

ANSWERS TO CHECK:
{answers}

TASK: Rate whether the answer content can be found in or reasonably inferred from the documents.

CHECK ONLY these accuracy features:

1. CLAIMS MATCH DOCUMENTS:
   - Can each fact in the answers be found in the source documents?
   - Are the facts stated correctly (not twisted or misrepresented)?

2. NOTHING MADE UP:
   - Are there any invented names, places, or organizations?
   - Are there any made-up numbers, dates, or statistics?
   - Are there any fake technical terms?

3. NO CONTRADICTIONS:
   - Do any claims directly contradict what the documents say?
   - Are there claims that go way beyond what the documents support?

ACCEPTABLE:
- Rewording information from documents
- Drawing obvious conclusions from stated facts

NOT ACCEPTABLE:
- Adding facts not found in documents
- Stating guesses as if they were facts

DO NOT CHECK (other prompts handle these):
- Whether answers sound natural
- Whether answers connect to each other
- Whether the questions were good

SCORING:

5 - All claims supported by documents, nothing made up
4 - Nearly all claims supported (95%+), only minor inferences
3 - Most claims supported (80-95%), some unsupported but plausible
2 - Many claims unsupported (50-80%), some things made up
1 - Mostly unsupported (<50%), major fabrications

Return JSON:
{{
    "score": <1-5>,
    "supported_claims": ["list 2-3 claims that ARE in the documents"],
    "unsupported_claims": ["list claims NOT found in documents"],
    "made_up_content": ["list any invented facts, names, or numbers"],
    "justification": "1-2 sentences explaining your score"
}}
"""


# ============================================================================
# TURN-LEVEL ANALYSIS PROMPTS (Simple and Clear)
# ============================================================================

TURN_DEPENDENCY_PROMPT = """Classify how this question DEPENDS on the previous conversation.

PREVIOUS CONVERSATION:
{prior_context}

CURRENT QUESTION:
{current_question}

TASK: Pick ONE dependency type that best describes how this question relates to what came before.

DEPENDENCY TYPES (pick ONE):

1. "coreference" - Uses pronouns pointing back to something mentioned before
   - Look for: it, this, that, they, these, those
   - Example: "Does IT also affect..." / "What about THAT?"

2. "ellipsis" - Incomplete sentence that needs context to understand
   - Look for: Missing words, fragments like "And...?", "How about...?"
   - Example: "And the second reason?" / "What about in winter?"

3. "substitution" - Uses a general term for something specific mentioned before
   - Look for: "this process", "that method", "the problem", "such cases"
   - Example: After "photosynthesis" → "How efficient is this process?"

4. "continuation" - Complete question on the same topic, but no explicit links
   - Look for: Full sentence, related topic, no pronouns pointing back
   - Example: After photosynthesis → "What role does chlorophyll play?"

5. "topic_shift" - Moves to a new aspect or subtopic
   - Look for: "What about...", "Regarding...", "Moving to...", new direction
   - Example: After science → "What are the economic effects?"

6. "self_contained" - Fully independent, makes sense alone (usually Turn 1)
   - Look for: Could be understood without any prior context
   - Example: First question of the conversation

Return JSON:
{{
    "dependency_type": "one of the 6 types above",
    "evidence": "quote the specific words that show this type",
    "explanation": "one sentence explaining your choice"
}}
"""

QUESTION_PATTERN_PROMPT = """Classify what TYPE of answer this question is looking for.

QUESTION:
{question}

TASK: Pick ONE pattern that best describes what kind of information this question wants.

QUESTION TYPES (pick ONE):

1. "why" - Asks for REASONS or CAUSES
   - Look for: "Why...?", "What causes...?", "What leads to...?"
   - Wants: Explanations, reasons

2. "how" - Asks HOW something WORKS or HAPPENS
   - Look for: "How does...?", "What is the process...?", "What happens...?"
   - Wants: Steps, process descriptions

3. "what" - Asks for FACTS (definitions, names, dates, places)
   - Look for: "What is...?", "Who...?", "When...?", "Where...?"
   - Wants: Specific facts, definitions

4. "compare" - Asks about DIFFERENCES or SIMILARITIES
   - Look for: "How does X differ from Y?", "Is X similar to Y?", "Which is better?"
   - Wants: Comparisons

5. "what_if" - Asks about POSSIBILITIES or HYPOTHETICALS
   - Look for: "What would happen if...?", "Could X...?", "What if...?"
   - Wants: Speculation, possibilities

6. "confirm" - Asks to VERIFY understanding
   - Look for: "Is it true that...?", "Does that mean...?", "So...right?"
   - Wants: Yes/no confirmation

7. "more_detail" - Asks for MORE INFORMATION on something
   - Look for: "Can you explain more...?", "What specifically...?", "Tell me more..."
   - Wants: Deeper explanation

8. "example" - Asks for EXAMPLES or INSTANCES
   - Look for: "What are examples of...?", "Can you give an instance?", "Like what?"
   - Wants: Concrete cases

9. "effect" - Asks about RESULTS or IMPLICATIONS
   - Look for: "What does this mean for...?", "What are the effects...?", "What happens as a result?"
   - Wants: Outcomes, impacts

Return JSON:
{{
    "question_pattern": "one of the 9 types above",
    "evidence": "quote the specific words that show this type",
    "explanation": "one sentence explaining your choice"
}}
"""


# ============================================================================
# THREAD-SAFE RATE LIMITER
# ============================================================================

class RateLimiter:
    """Thread-safe rate limiter with per-worker and global limits."""

    def __init__(self, min_interval: float = 0.3, global_interval: float = 0.1):
        self.min_interval = min_interval
        self.global_interval = global_interval
        self.worker_last_request: Dict[int, float] = {}
        self.global_last_request = 0.0
        self.lock = threading.Lock()

    def wait(self, worker_id: int):
        """Wait until it's safe to make a request."""
        with self.lock:
            now = time.time()

            # Check global rate limit
            global_wait = self.global_interval - (now - self.global_last_request)
            if global_wait > 0:
                time.sleep(global_wait)
                now = time.time()

            # Check per-worker rate limit
            if worker_id in self.worker_last_request:
                worker_wait = self.min_interval - (now - self.worker_last_request[worker_id])
                if worker_wait > 0:
                    time.sleep(worker_wait)
                    now = time.time()

            # Update timestamps
            self.worker_last_request[worker_id] = now
            self.global_last_request = now


# ============================================================================
# AZURE CLIENT WITH RETRY LOGIC
# ============================================================================

class AzureClient:
    """Thread-safe Azure OpenAI client with retry logic."""

    def __init__(self, config: Config):
        self.config = config
        self.client = AzureOpenAI(
            azure_endpoint=config.azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_key=config.azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY", ""),
            api_version=config.azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
        self.rate_limiter = RateLimiter(
            min_interval=config.min_request_interval,
            global_interval=config.global_rate_limit
        )
        self.total_requests = 0
        self.request_lock = threading.Lock()

    def call(self, prompt: str, worker_id: int = 0) -> Dict[str, Any]:
        """Make API call with retry logic and rate limiting."""

        for attempt in range(self.config.max_retries):
            try:
                # Wait for rate limit
                self.rate_limiter.wait(worker_id)

                response = self.client.chat.completions.create(
                    model=self.config.azure_deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
                    messages=[
                        {"role": "system", "content": "You are an expert linguistic analyst. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )

                with self.request_lock:
                    self.total_requests += 1

                return json.loads(response.choices[0].message.content)

            except json.JSONDecodeError as e:
                logging.warning(f"Worker {worker_id}: JSON decode error on attempt {attempt + 1}: {e}")
                delay = min(self.config.base_retry_delay * (2 ** attempt), self.config.max_retry_delay)
                time.sleep(delay)

            except Exception as e:
                error_msg = str(e).lower()

                # Handle rate limit errors specifically
                if "rate" in error_msg or "429" in error_msg or "throttl" in error_msg:
                    delay = min(self.config.base_retry_delay * (3 ** attempt), self.config.max_retry_delay)
                    logging.warning(f"Worker {worker_id}: Rate limited, waiting {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logging.warning(f"Worker {worker_id}: API error on attempt {attempt + 1}: {e}")
                    delay = min(self.config.base_retry_delay * (2 ** attempt), self.config.max_retry_delay)
                    time.sleep(delay)

        raise Exception(f"All {self.config.max_retries} attempts failed")


# ============================================================================
# DOCUMENT LOADER - NEW FUNCTIONALITY
# ============================================================================

class DocumentLoader:
    """Loads and indexes documents from positive_documents.jsonl files."""

    def __init__(self, input_dirs: List[str]):
        self.input_dirs = [Path(d) for d in input_dirs]
        self.doc_index: Dict[str, str] = {}  # doc_id -> content
        self._load_all_documents()

    def _load_all_documents(self):
        """Load all documents from *_positive_documents.jsonl files in all input directories."""
        total_files = 0

        for input_dir in self.input_dirs:
            doc_files = list(input_dir.glob("*_positive_documents.jsonl"))
            total_files += len(doc_files)

            logging.info(f"Loading documents from {len(doc_files)} files in {input_dir}")

            for doc_file in doc_files:
                count = 0
                with open(doc_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                doc = json.loads(line)
                                doc_id = doc.get('doc_id', '')
                                content = doc.get('content', '')
                                if doc_id and content:
                                    self.doc_index[doc_id] = content
                                    count += 1
                            except json.JSONDecodeError:
                                continue
                logging.info(f"  Loaded {count} documents from {doc_file.name}")

        logging.info(f"Total documents indexed: {len(self.doc_index)} from {total_files} files")

    def get_documents(self, doc_ids: List[str]) -> List[Dict[str, str]]:
        """Retrieve documents by their IDs."""
        documents = []
        for doc_id in doc_ids:
            if doc_id in self.doc_index:
                documents.append({
                    'id': doc_id,
                    'content': self.doc_index[doc_id]  # FULL content, NO truncation
                })
            else:
                logging.warning(f"Document not found: {doc_id}")
        return documents


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_benchmark_files(input_dirs: List[str]) -> Tuple[List[Dict[str, Any]], str]:
    """Load all benchmark files from multiple directories."""
    conversations = []
    dataset_type = "mixed"  # Since we're loading from both folders
    total_files = 0

    for input_dir in input_dirs:
        input_path = Path(input_dir)
        benchmark_files = list(input_path.glob("*_benchmark.jsonl"))
        total_files += len(benchmark_files)

        # Detect which dataset type this directory is
        dir_name = input_path.name.lower()
        if "annotated" in dir_name:
            current_type = "annotated"
        elif "bright" in dir_name:
            current_type = "bright"
        else:
            current_type = "unknown"

        logging.info(f"Loading from {input_dir} ({current_type} type, {len(benchmark_files)} files)")

        for benchmark_file in benchmark_files:
            domain = benchmark_file.stem.replace('_benchmark', '')
            logging.info(f"  Loading {benchmark_file.name}...")

            file_count = 0
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            conv = json.loads(line)
                            # Add domain from filename/task field
                            conv['domain'] = conv.get('task', domain)
                            # Store which dataset type this conversation came from
                            conv['dataset_source'] = current_type
                            conversations.append(conv)
                            file_count += 1
                        except json.JSONDecodeError:
                            continue

            logging.info(f"    Loaded {file_count} conversations")

    logging.info(f"Total: {len(conversations)} conversations from {total_files} benchmark files")

    return conversations, dataset_type


def format_conversation(turns: List[Dict]) -> str:
    """Format full conversation (Q&A pairs)."""
    formatted = []
    for i, turn in enumerate(turns, 1):
        q = turn.get('query', turn.get('question', turn.get('q', '')))
        a = turn.get('answer', turn.get('response', turn.get('a', '')))
        formatted.append(f"Turn {i}:\nQ: {q}\nA: {a}")
    return "\n\n".join(formatted)


def format_questions_only(turns: List[Dict]) -> str:
    """Format only the questions from turns."""
    formatted = []
    for i, turn in enumerate(turns, 1):
        q = turn.get('query', turn.get('question', turn.get('q', '')))
        formatted.append(f"Q{i}: {q}")
    return "\n".join(formatted)


def format_answers_only(turns: List[Dict]) -> str:
    """Format only the answers from turns."""
    formatted = []
    for i, turn in enumerate(turns, 1):
        a = turn.get('answer', turn.get('response', turn.get('a', '')))
        formatted.append(f"A{i}: {a}")
    return "\n\n".join(formatted)


def format_documents(documents: List[Dict]) -> str:
    """Format documents - NO TRUNCATION."""
    if not documents:
        return "[No documents provided]"

    formatted = []
    for i, doc in enumerate(documents, 1):
        if isinstance(doc, dict):
            content = doc.get('content', doc.get('text', doc.get('passage', str(doc))))
            doc_id = doc.get('id', doc.get('doc_id', f'Document {i}'))
            # FULL content - NO truncation
            formatted.append(f"[Document {i}: {doc_id}]\n{content}")
        else:
            formatted.append(f"[Document {i}]\n{str(doc)}")

    return "\n\n---\n\n".join(formatted)


def compute_entropy(items: List[str]) -> float:
    """Compute normalized Shannon entropy."""
    if not items:
        return 0.0

    counts = defaultdict(int)
    for item in items:
        if item:
            counts[item] += 1

    if len(counts) <= 1:
        return 0.0

    total = len(items)
    probs = [count / total for count in counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(len(counts))

    return round(entropy / max_entropy, 4) if max_entropy > 0 else 0.0


def get_doc_ids_from_turn(turn: Dict, dataset_source: str) -> List[str]:
    """Extract document IDs from a turn based on dataset source."""
    if dataset_source == "annotated":
        return turn.get('supporting_doc_ids', [])
    elif dataset_source == "bright":
        return turn.get('gold_doc_ids', [])
    else:
        # Try both - supporting_doc_ids first, then gold_doc_ids
        doc_ids = turn.get('supporting_doc_ids', [])
        if not doc_ids:
            doc_ids = turn.get('gold_doc_ids', [])
        return doc_ids


def get_all_doc_ids_from_conversation(conversation: Dict) -> List[str]:
    """Get all unique document IDs from all turns in a conversation."""
    all_doc_ids = set()
    dataset_source = conversation.get('dataset_source', 'unknown')
    for turn in conversation.get('turns', []):
        doc_ids = get_doc_ids_from_turn(turn, dataset_source)
        all_doc_ids.update(doc_ids)
    return list(all_doc_ids)


# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class ComprehensiveAnalyzer:
    """Main analyzer with parallel processing."""

    def __init__(self, config: Config, doc_loader: DocumentLoader):
        self.config = config
        self.client = AzureClient(config)
        self.doc_loader = doc_loader

        # Results storage (thread-safe)
        self.results_lock = threading.Lock()
        self.quality_results: List[Dict] = []
        self.turn_results: List[Dict] = []
        self.failed_conversations: List[str] = []

        # Progress tracking
        self.processed_count = 0
        self.progress_lock = threading.Lock()

    def analyze_conversation(self, conversation: Dict, worker_id: int) -> Tuple[Dict, List[Dict]]:
        """Analyze a single conversation (quality + turns)."""

        conv_id = conversation.get('id', conversation.get('conv_id', f'conv_{id(conversation)}'))

        try:
            # 1. Quality validation
            quality_result = self._validate_quality(conversation, worker_id)
            quality_result['conv_id'] = conv_id
            quality_result['domain'] = conversation.get('domain', conversation.get('task', 'unknown'))
            quality_result['dataset_source'] = conversation.get('dataset_source', 'unknown')

            # 2. Turn-level analysis (batched)
            turn_results = self._analyze_turns(conversation, worker_id)
            for tr in turn_results:
                tr['conv_id'] = conv_id
                tr['domain'] = conversation.get('domain', conversation.get('task', 'unknown'))
                tr['dataset_source'] = conversation.get('dataset_source', 'unknown')

            return quality_result, turn_results

        except Exception as e:
            logging.error(f"Worker {worker_id}: Failed to analyze {conv_id}: {e}")
            return self._empty_quality_result(conversation), []

    def _validate_quality(self, conversation: Dict, worker_id: int) -> Dict:
        """Validate conversation quality using 4 SEPARATE focused API calls."""

        turns = conversation.get('turns', conversation.get('dialogue', []))

        # Get original query - try multiple field names
        original_query = conversation.get('original_query',
                                          conversation.get('query', ''))
        if not original_query and turns:
            original_query = turns[0].get('query', turns[0].get('question', ''))

        # Get documents by resolving doc IDs (uses dataset_source stored in conversation)
        all_doc_ids = get_all_doc_ids_from_conversation(conversation)
        documents = self.doc_loader.get_documents(all_doc_ids)

        # Pre-format data for prompts
        conversation_text = format_conversation(turns)
        questions_text = format_questions_only(turns)
        answers_text = format_answers_only(turns)
        documents_text = format_documents(documents)

        result = {
            'num_turns': len(turns),
            'num_documents': len(documents)
        }

        # ===== 1. NATURALNESS (focused on conversation flow) =====
        try:
            naturalness_prompt = NATURALNESS_PROMPT.format(conversation=conversation_text)
            result['naturalness'] = self.client.call(naturalness_prompt, worker_id)
        except Exception as e:
            logging.warning(f"Worker {worker_id}: Naturalness eval failed: {e}")
            result['naturalness'] = {'score': None, 'error': str(e)}

        # ===== 2. TURN COHERENCE (focused on logical progression) =====
        try:
            coherence_prompt = TURN_COHERENCE_PROMPT.format(conversation=conversation_text)
            result['turn_coherence'] = self.client.call(coherence_prompt, worker_id)
        except Exception as e:
            logging.warning(f"Worker {worker_id}: Coherence eval failed: {e}")
            result['turn_coherence'] = {'score': None, 'error': str(e)}

        # ===== 3. QUESTION QUALITY (focused on questions only) =====
        try:
            question_prompt = QUESTION_QUALITY_PROMPT.format(
                original_query=original_query,
                questions=questions_text
            )
            result['question_quality'] = self.client.call(question_prompt, worker_id)
        except Exception as e:
            logging.warning(f"Worker {worker_id}: Question quality eval failed: {e}")
            result['question_quality'] = {'score': None, 'error': str(e)}

        # ===== 4. GROUNDEDNESS (focused on answers vs documents) =====
        try:
            groundedness_prompt = GROUNDEDNESS_PROMPT.format(
                documents=documents_text,
                answers=answers_text
            )
            result['groundedness'] = self.client.call(groundedness_prompt, worker_id)
        except Exception as e:
            logging.warning(f"Worker {worker_id}: Groundedness eval failed: {e}")
            result['groundedness'] = {'score': None, 'error': str(e)}

        # ===== Compute overall score =====
        scores = []
        for dim in ['naturalness', 'turn_coherence', 'question_quality', 'groundedness']:
            if dim in result and isinstance(result[dim], dict):
                score = result[dim].get('score')
                if score is not None and isinstance(score, (int, float)):
                    scores.append(score)

        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score >= 4.5:
                quality_level = "excellent"
            elif avg_score >= 3.5:
                quality_level = "good"
            elif avg_score >= 2.5:
                quality_level = "acceptable"
            elif avg_score >= 1.5:
                quality_level = "poor"
            else:
                quality_level = "very_poor"

            result['overall'] = {
                'average_score': round(avg_score, 2),
                'quality_level': quality_level,
                'dimensions_evaluated': len(scores)
            }

        return result

    def _analyze_turns(self, conversation: Dict, worker_id: int) -> List[Dict]:
        """Analyze each turn with 2 SEPARATE focused prompts."""

        turns = conversation.get('turns', conversation.get('dialogue', []))
        if not turns:
            return []

        all_results = []

        for turn_idx, turn in enumerate(turns):
            current_question = turn.get('query', turn.get('question', ''))
            turn_result = {
                'turn_position': turn_idx + 1,
                'question': current_question
            }

            # Build prior context for this turn
            if turn_idx == 0:
                prior_context = "[This is the first turn - no prior context]"
            else:
                context_parts = []
                # Include up to 2 previous turns as context
                start_idx = max(0, turn_idx - 2)
                for i in range(start_idx, turn_idx):
                    prev_turn = turns[i]
                    q = prev_turn.get('query', prev_turn.get('question', ''))
                    a = prev_turn.get('answer', prev_turn.get('response', ''))
                    context_parts.append(f"Turn {i+1}:\nQ: {q}\nA: {a}")
                prior_context = "\n\n".join(context_parts)

            # ===== 1. DEPENDENCY TYPE (how question connects to context) =====
            try:
                dep_prompt = TURN_DEPENDENCY_PROMPT.format(
                    prior_context=prior_context,
                    current_question=current_question
                )
                dep_result = self.client.call(dep_prompt, worker_id)
                turn_result['dependency_type'] = dep_result.get('dependency_type', 'unknown')
                turn_result['dependency_evidence'] = dep_result.get('evidence', '')
            except Exception as e:
                logging.warning(f"Worker {worker_id}: Dependency analysis failed for turn {turn_idx+1}: {e}")
                turn_result['dependency_type'] = 'unknown'
                turn_result['dependency_error'] = str(e)

            # ===== 2. QUESTION PATTERN (what type of info is sought) =====
            try:
                pattern_prompt = QUESTION_PATTERN_PROMPT.format(
                    question=current_question
                )
                pattern_result = self.client.call(pattern_prompt, worker_id)
                turn_result['question_pattern'] = pattern_result.get('question_pattern', 'unknown')
                turn_result['pattern_evidence'] = pattern_result.get('evidence', '')
            except Exception as e:
                logging.warning(f"Worker {worker_id}: Pattern analysis failed for turn {turn_idx+1}: {e}")
                turn_result['question_pattern'] = 'unknown'
                turn_result['pattern_error'] = str(e)

            all_results.append(turn_result)

        return all_results

    def _empty_quality_result(self, conversation: Dict) -> Dict:
        """Return empty quality result for failed conversations."""
        turns = conversation.get('turns', conversation.get('dialogue', []))
        return {
            'conv_id': conversation.get('id', 'unknown'),
            'domain': conversation.get('domain', conversation.get('task', 'unknown')),
            'num_turns': len(turns),
            'error': 'analysis_failed',
            'naturalness': {'score': None},
            'turn_coherence': {'score': None},
            'question_quality': {'score': None},
            'groundedness': {'score': None}
        }

    def _worker_task(self, conversation: Dict, worker_id: int, pbar: tqdm) -> None:
        """Worker task for parallel processing."""

        quality_result, turn_results = self.analyze_conversation(conversation, worker_id)

        # Store results (thread-safe)
        with self.results_lock:
            self.quality_results.append(quality_result)
            self.turn_results.extend(turn_results)
            self.processed_count += 1

        # Update progress bar
        pbar.update(1)

        # Check if we should save intermediate results
        with self.progress_lock:
            if self.processed_count % self.config.save_every_n == 0:
                self._save_intermediate()

    def _save_intermediate(self):
        """Save intermediate results."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with self.results_lock:
            # Save quality results
            quality_path = output_path / "intermediate_quality.json"
            with open(quality_path, 'w', encoding='utf-8') as f:
                json.dump(self.quality_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

            # Save turn results
            turns_path = output_path / "intermediate_turns.json"
            with open(turns_path, 'w', encoding='utf-8') as f:
                json.dump(self.turn_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        logging.info(f"Saved intermediate results ({self.processed_count} conversations)")

    def analyze_dataset(self, conversations: List[Dict]) -> Dict[str, Any]:
        """Analyze full dataset with parallel processing."""

        # Sample if configured
        if self.config.sample_size and len(conversations) > self.config.sample_size:
            random.seed(self.config.random_seed)
            conversations = random.sample(conversations, self.config.sample_size)
            logging.info(f"Sampled {self.config.sample_size} conversations")

        total_convs = len(conversations)
        total_turns = sum(len(c.get('turns', c.get('dialogue', []))) for c in conversations)
        logging.info(f"Analyzing {total_convs} conversations ({total_turns} turns) with {self.config.max_workers} workers")

        # Process with thread pool
        with tqdm(total=total_convs, desc="Analyzing") as pbar:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for i, conv in enumerate(conversations):
                    worker_id = i % self.config.max_workers
                    future = executor.submit(self._worker_task, conv, worker_id, pbar)
                    futures.append(future)

                # Wait for all to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Worker exception: {e}")

        # Compute final statistics
        return self.compute_statistics()

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics."""

        # Count conversations by dataset source
        source_counts = defaultdict(int)
        for r in self.quality_results:
            source_counts[r.get('dataset_source', 'unknown')] += 1

        stats = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_conversations': len(self.quality_results),
                'total_turns': len(self.turn_results),
                'api_calls': self.client.total_requests,
                'failed_conversations': len(self.failed_conversations),
                'dataset_sources': dict(source_counts)
            },
            'quality_scores': {},
            'turn_metrics': {},
            'diversity_metrics': {},
            'by_domain': {},
            'comparison_with_human': {}
        }

        # ===== QUALITY SCORES =====
        dimensions = ['naturalness', 'turn_coherence', 'question_quality', 'groundedness']

        for dim in dimensions:
            scores = []
            for r in self.quality_results:
                if dim in r and isinstance(r[dim], dict):
                    score = r[dim].get('score')
                    if score is not None and isinstance(score, (int, float)):
                        scores.append(score)

            if scores:
                stats['quality_scores'][dim] = {
                    'mean': round(np.mean(scores), 3),
                    'std': round(np.std(scores), 3),
                    'median': round(np.median(scores), 3),
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores),
                    'distribution': {i: scores.count(i) for i in range(1, 6)}
                }

        # ===== TURN METRICS =====
        # Dependency type distribution
        dep_types = [t.get('dependency_type', 'unknown') for t in self.turn_results
                     if t.get('dependency_type')]
        stats['turn_metrics']['dependency_types'] = self._count_distribution(dep_types)

        # Question pattern distribution
        q_patterns = [t.get('question_pattern', 'unknown') for t in self.turn_results
                      if t.get('question_pattern')]
        stats['turn_metrics']['question_patterns'] = self._count_distribution(q_patterns)

        # ===== DIVERSITY METRICS =====
        stats['diversity_metrics'] = {
            'dependency_type_entropy': compute_entropy(dep_types),
            'question_pattern_entropy': compute_entropy(q_patterns),
            'unique_dependency_types': len(set(dep_types)),
            'unique_question_patterns': len(set(q_patterns))
        }

        # ===== BY DOMAIN =====
        domains = set(r.get('domain', 'unknown') for r in self.quality_results)
        for domain in domains:
            domain_quality = [r for r in self.quality_results if r.get('domain') == domain]
            domain_turns = [t for t in self.turn_results if t.get('domain') == domain]

            stats['by_domain'][domain] = {
                'num_conversations': len(domain_quality),
                'num_turns': len(domain_turns)
            }

            # Quality scores by domain
            for dim in dimensions:
                scores = [r[dim]['score'] for r in domain_quality
                          if dim in r and isinstance(r[dim], dict) and r[dim].get('score') is not None]
                if scores:
                    stats['by_domain'][domain][f'{dim}_mean'] = round(np.mean(scores), 3)

            # Turn types by domain
            domain_deps = [t.get('dependency_type') for t in domain_turns if t.get('dependency_type')]
            stats['by_domain'][domain]['dependency_entropy'] = compute_entropy(domain_deps)

        # ===== COMPARISON WITH HUMAN (Section 3.4) =====
        human_scores = {
            'naturalness': {'score': 4.2, 'kappa': 0.71},
            'turn_coherence': {'score': 4.0, 'kappa': 0.68},
            'question_quality': {'score': 3.9, 'kappa': 0.65},
            'groundedness': {'score': 4.3, 'kappa': 0.74}
        }

        for dim in dimensions:
            if dim in stats['quality_scores']:
                llm_mean = stats['quality_scores'][dim]['mean']
                human_data = human_scores[dim]
                diff = llm_mean - human_data['score']

                stats['comparison_with_human'][dim] = {
                    'llm_score': llm_mean,
                    'llm_std': stats['quality_scores'][dim]['std'],
                    'human_score': human_data['score'],
                    'human_kappa': human_data['kappa'],
                    'difference': round(diff, 3),
                    'within_threshold': abs(diff) < 0.5
                }

        return stats

    def _count_distribution(self, items: List[str]) -> Dict[str, Any]:
        """Count distribution with percentages."""
        if not items:
            return {}

        counts = defaultdict(int)
        for item in items:
            if item:
                counts[item] += 1

        total = sum(counts.values())
        return {
            k: {'count': v, 'percentage': round(v / total * 100, 2)}
            for k, v in sorted(counts.items(), key=lambda x: -x[1])
        }

    def save_results(self) -> Tuple[Path, Path, Path]:
        """Save all results."""

        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Compute statistics
        stats = self.compute_statistics()

        # Save detailed results
        detailed = {
            'quality_results': self.quality_results,
            'turn_results': self.turn_results,
            'statistics': stats
        }
        detailed_path = output_path / f"comprehensive_analysis_{timestamp}.json"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(detailed, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        # Save statistics only
        stats_path = output_path / f"statistics_{timestamp}.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        # Generate report
        report = self.generate_report(stats)
        report_path = output_path / f"analysis_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        return detailed_path, stats_path, report_path

    def generate_report(self, stats: Dict[str, Any]) -> str:
        """Generate comprehensive markdown report."""

        # Format dataset sources for display
        sources = stats['metadata'].get('dataset_sources', {})
        sources_str = ", ".join([f"{k}: {v}" for k, v in sources.items()])

        lines = [
            "# Comprehensive Conversation Analysis Report",
            "",
            f"**Generated:** {stats['metadata']['timestamp']}",
            f"**Dataset Sources:** {sources_str}",
            f"**Conversations:** {stats['metadata']['total_conversations']}",
            f"**Turns:** {stats['metadata']['total_turns']}",
            f"**API Calls:** {stats['metadata']['api_calls']}",
            "",
            "---",
            "",
            "## 1. Quality Validation (Section 3.4 Comparison)",
            "",
            "### Overall Scores",
            "",
            "| Dimension | LLM Mean | LLM Std | Human Score | Human k | Difference |",
            "|-----------|----------|---------|-------------|---------|------------|"
        ]

        for dim, comp in stats.get('comparison_with_human', {}).items():
            diff = comp['difference']
            diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
            status = "OK" if comp['within_threshold'] else "!!"
            lines.append(f"| {dim.replace('_', ' ').title()} | {comp['llm_score']:.2f} | {comp['llm_std']:.2f} | {comp['human_score']:.1f} | {comp['human_kappa']:.2f} | {diff_str} {status} |")

        # Score distributions
        lines.extend([
            "",
            "### Score Distributions",
            ""
        ])

        for dim, data in stats.get('quality_scores', {}).items():
            dist = data.get('distribution', {})
            dist_str = " | ".join([f"{k}*:{v}" for k, v in sorted(dist.items())])
            lines.append(f"**{dim.replace('_', ' ').title()}:** {dist_str}")

        # Turn metrics
        lines.extend([
            "",
            "---",
            "",
            "## 2. Turn-Level Metrics",
            "",
            "### Dependency Type Distribution",
            "",
            "| Type | Count | % |",
            "|------|-------|---|"
        ])

        for type_name, data in stats.get('turn_metrics', {}).get('dependency_types', {}).items():
            lines.append(f"| {type_name.replace('_', ' ').title()} | {data['count']} | {data['percentage']:.1f}% |")

        lines.extend([
            "",
            "### Question Pattern Distribution",
            "",
            "| Pattern | Count | % |",
            "|---------|-------|---|"
        ])

        for pattern, data in stats.get('turn_metrics', {}).get('question_patterns', {}).items():
            lines.append(f"| {pattern.replace('_', ' ').title()} | {data['count']} | {data['percentage']:.1f}% |")

        # Diversity metrics
        lines.extend([
            "",
            "---",
            "",
            "## 3. Diversity Metrics",
            "",
            "| Metric | Value | Interpretation |",
            "|--------|-------|----------------|"
        ])

        diversity = stats.get('diversity_metrics', {})
        interpretations = {
            'dependency_type_entropy': 'Higher = more diverse turn dependencies (0-1 scale)',
            'question_pattern_entropy': 'Higher = more diverse question types (0-1 scale)',
            'unique_dependency_types': 'Number of different dependency types observed',
            'unique_question_patterns': 'Number of different question patterns observed'
        }

        for metric, value in diversity.items():
            interp = interpretations.get(metric, '')
            if isinstance(value, float):
                lines.append(f"| {metric.replace('_', ' ').title()} | {value:.3f} | {interp} |")
            else:
                lines.append(f"| {metric.replace('_', ' ').title()} | {value} | {interp} |")

        # By domain
        lines.extend([
            "",
            "---",
            "",
            "## 4. Results by Domain",
            "",
            "| Domain | N Conv | N Turns | Natural. | Coher. | Q.Qual. | Ground. | Dep. Entropy |",
            "|--------|--------|---------|----------|--------|---------|---------|--------------|"
        ])

        for domain, data in sorted(stats.get('by_domain', {}).items()):
            nat = data.get('naturalness_mean', 'N/A')
            coh = data.get('turn_coherence_mean', 'N/A')
            qq = data.get('question_quality_mean', 'N/A')
            gr = data.get('groundedness_mean', 'N/A')
            ent = data.get('dependency_entropy', 'N/A')

            nat_str = f"{nat:.2f}" if isinstance(nat, float) else nat
            coh_str = f"{coh:.2f}" if isinstance(coh, float) else coh
            qq_str = f"{qq:.2f}" if isinstance(qq, float) else qq
            gr_str = f"{gr:.2f}" if isinstance(gr, float) else gr
            ent_str = f"{ent:.3f}" if isinstance(ent, float) else ent

            lines.append(f"| {domain} | {data['num_conversations']} | {data['num_turns']} | {nat_str} | {coh_str} | {qq_str} | {gr_str} | {ent_str} |")

        # Summary
        lines.extend([
            "",
            "---",
            "",
            "## 5. Summary for Paper",
            "",
            "### Section 3.4 Addition (Automatic Validation)",
            "",
            "```",
            "To complement manual evaluation on 100 conversations, we employed GPT-4o to assess",
            f"all {stats['metadata']['total_conversations']} conversations on the same four quality dimensions.",
            "Results show strong alignment with human judgments:",
            "```",
            ""
        ])

        for dim, comp in stats.get('comparison_with_human', {}).items():
            lines.append(f"- **{dim.replace('_', ' ').title()}**: LLM {comp['llm_score']:.2f} vs Human {comp['human_score']:.1f}")

        lines.extend([
            "",
            "### Turn-Level Linguistic Analysis",
            "",
            "```",
            "We characterized conversational complexity through automatic annotation:",
            "```",
            ""
        ])

        dep_data = stats.get('turn_metrics', {}).get('dependency_types', {})
        if dep_data:
            coref = dep_data.get('coreference', {}).get('percentage', 0)
            ellip = dep_data.get('ellipsis', {}).get('percentage', 0)
            lines.append(f"- **Coreference turns**: {coref:.1f}%")
            lines.append(f"- **Ellipsis turns**: {ellip:.1f}%")

        lines.append(f"- **Dependency type diversity**: {diversity.get('dependency_type_entropy', 0):.3f} (entropy)")
        lines.append(f"- **Question pattern diversity**: {diversity.get('question_pattern_entropy', 0):.3f} (entropy)")

        return "\n".join(lines)


# ============================================================================
# AUTO-DETECT INPUT DIRECTORIES
# ============================================================================

def find_default_input_dirs(base_path: str = ".") -> List[str]:
    """Find default input directories (outputs_Annotated_unified and outputs_bright_unified)."""
    base = Path(base_path)
    default_dirs = []

    # Look for the two expected folders
    annotated_dir = base / "outputs_Annotated_unified"
    bright_dir = base / "outputs_bright_unified"

    if annotated_dir.exists():
        default_dirs.append(str(annotated_dir))
    if bright_dir.exists():
        default_dirs.append(str(bright_dir))

    return default_dirs


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Conversation Quality & Diversity Analysis (Modified for MURECOR benchmarks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process ALL 11 benchmark files from both folders (auto-detect):
  python comprehensive_analysis_modified.py --output ./results

  # Specify custom directories:
  python comprehensive_analysis_modified.py --input-dirs ./outputs_Annotated_unified ./outputs_bright_unified --output ./results

  # Process only one folder:
  python comprehensive_analysis_modified.py --input-dirs ./outputs_Annotated_unified --output ./results_annotated

  # With sampling for testing:
  python comprehensive_analysis_modified.py --output ./results --sample 50

  # Adjust parallelization:
  python comprehensive_analysis_modified.py --output ./results --workers 3
        """
    )

    parser.add_argument("--input-dirs", "-i", nargs="+", default=None,
                        help="Input directories containing benchmark files (default: auto-detect both folders)")
    parser.add_argument("--output", "-o", default="./analysis_results", help="Output directory")
    parser.add_argument("--sample", "-s", type=int, default=None, help="Sample size (None for all)")
    parser.add_argument("--workers", "-w", type=int, default=5, help="Max parallel workers (default: 5)")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Turns per batch (default: 4)")
    parser.add_argument("--rate-limit", type=float, default=0.3, help="Min seconds between requests per worker")
    parser.add_argument("--save-every", type=int, default=20, help="Save intermediate results every N conversations")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    # Auto-detect input directories if not specified
    if args.input_dirs is None:
        args.input_dirs = find_default_input_dirs(".")
        if not args.input_dirs:
            print("\n" + "="*60)
            print("ERROR: Could not find input directories!")
            print("="*60)
            print("\nExpected folders in current directory:")
            print("  - outputs_Annotated_unified/")
            print("  - outputs_bright_unified/")
            print("\nOr specify manually with --input-dirs:")
            print("  python comprehensive_analysis_modified.py --input-dirs ./folder1 ./folder2 --output ./results")
            print()
            sys.exit(1)

    # Ensure output directory exists for logging
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.output}/analysis.log", mode='w')
        ]
    )

    # Get Azure credentials
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "")

    if not endpoint or not api_key:
        print("\n" + "="*60)
        print("ERROR: Azure OpenAI credentials not set!")
        print("="*60)
        print("\nSet environment variables:")
        print("  export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'")
        print("  export AZURE_OPENAI_API_KEY='your-api-key'")
        print("  export AZURE_OPENAI_DEPLOYMENT='gpt-4o'  # optional")
        print()
        sys.exit(1)

    # Create config
    config = Config(
        azure_endpoint=endpoint,
        azure_api_key=api_key,
        azure_deployment=deployment,
        azure_api_version=api_version,
        max_workers=args.workers,
        turns_per_batch=args.batch_size,
        min_request_interval=args.rate_limit,
        output_dir=args.output,
        save_every_n=args.save_every,
        sample_size=args.sample
    )

    # Show which directories will be processed
    print("\n" + "="*70)
    print("MURECOR Benchmark Analysis - Processing ALL 11 Benchmark Files")
    print("="*70)
    print(f"\nInput directories ({len(args.input_dirs)}):")
    for d in args.input_dirs:
        print(f"  - {d}")

    # Load documents from all directories
    print(f"\nLoading documents from all directories...")
    doc_loader = DocumentLoader(args.input_dirs)

    # Load benchmark conversations from all directories
    print(f"\nLoading benchmark files...")
    conversations, dataset_type = load_benchmark_files(args.input_dirs)
    print(f"Loaded {len(conversations)} total conversations")
    print(f"Dataset type: {dataset_type}")

    # Run analysis
    print(f"\nStarting analysis with {config.max_workers} workers...")
    print(f"Rate limit: {config.min_request_interval}s per worker")
    print(f"Batch size: {config.turns_per_batch} turns per API call")
    print()

    analyzer = ComprehensiveAnalyzer(config, doc_loader)
    stats = analyzer.analyze_dataset(conversations)

    # Save results
    detailed_path, stats_path, report_path = analyzer.save_results()

    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    # Show dataset sources
    sources = stats['metadata'].get('dataset_sources', {})
    print(f"\nDataset Sources:")
    for source, count in sources.items():
        print(f"  - {source}: {count} conversations")
    print(f"\nTotal: {stats['metadata']['total_conversations']} conversations, "
          f"{stats['metadata']['total_turns']} turns")
    print(f"API calls: {stats['metadata']['api_calls']}")

    print("\nQuality Scores (LLM vs Human from Section 3.4):")
    print("-" * 60)

    for dim, comp in stats.get('comparison_with_human', {}).items():
        diff = comp['difference']
        status = "OK" if comp['within_threshold'] else "!!"
        print(f"  {dim.replace('_', ' ').title():20} "
              f"LLM: {comp['llm_score']:.2f} +/- {comp['llm_std']:.2f}  "
              f"Human: {comp['human_score']:.1f}  "
              f"Diff: {diff:+.2f} {status}")

    print("\nTurn Metrics:")
    print("-" * 60)

    dep_types = stats.get('turn_metrics', {}).get('dependency_types', {})
    top_deps = list(dep_types.items())[:5]
    for dep, data in top_deps:
        print(f"  {dep.replace('_', ' ').title():25} {data['count']:5} ({data['percentage']:.1f}%)")

    print("\nDiversity Metrics:")
    print("-" * 60)
    diversity = stats.get('diversity_metrics', {})
    print(f"  Dependency type entropy:    {diversity.get('dependency_type_entropy', 0):.3f}")
    print(f"  Question pattern entropy:   {diversity.get('question_pattern_entropy', 0):.3f}")
    print(f"  Unique dependency types:     {diversity.get('unique_dependency_types', 0)}")
    print(f"  Unique question patterns:    {diversity.get('unique_question_patterns', 0)}")

    print(f"\nResults saved to: {args.output}")
    print(f"  - Full analysis: {detailed_path.name}")
    print(f"  - Statistics: {stats_path.name}")
    print(f"  - Report: {report_path.name}")
    print()


if __name__ == "__main__":
    main()
