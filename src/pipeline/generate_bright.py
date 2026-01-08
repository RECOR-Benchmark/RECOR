import os
import json
import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

from datasets import load_dataset
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CENTRALIZED CONFIGURATION (MODIFIED FOR DEPTH)
# ============================================================

CONFIG = {
    "aspects": {
        "min_count": 4,
        "max_count": {
            "batch": 12,
            "single": 5
        },
        "document_coverage_min": 0.50,
        "max_extraction_attempts": 4,
        "overlap_confidence_threshold": 0.8
    },
    
    "batch": {
        "extraction_attempts": 6,
        "aspects_per_attempt": 5
    },
    
    "subquestions": {
        "max_generation_retries": 4,
        "retry_temperature_boost": 0.1
    },
    
    "turns": {
        "min_count": 3,
        "min_answerability": 0.40,
        "diversity": {
            "max_similarity": 0.58,
            "min_new_words": 10,
            "max_retries": 1,
            "semantic_fact_overlap_threshold": 0.70,
            "max_repeated_phrases": 1,
            "max_phrase_repetition_ratio": 0.30
        },
        "history_summarization_threshold": 5
    },
    
    "documents": {
        "scoring_threshold": 4.0,
        "fallback_threshold": 6.0,
        "per_turn_default": 5,
        "truncation_limit": 800
    },
    
    "llm": {
        "temperature": {
            "alignment": 0.3,
            "aspect_overlap": 0.2,
            "aspect_suitability": 0.2,
            "turn_diversity": 0.2,
            "aspect_extraction": 0.6,
            "semantic_facts": 0.4,
            "fact_verification": 0.2,
            "subquestion": 0.6,
            "reasoning": 0.5,
            "subquestion_reasoning": 0.4,
            "answerability": 0.2,
            "document_scoring": 0.2,
            "query_turn1": 0.5,
            "query_followup": 0.5,
            "answer": 0.6,
            "aspect_overlap_check": 0.2,
            "aspect_ordering": 0.2,
            "semantic_similarity": 0.2
        },
        "similarity": {
            "llm_skip_high": 0.70,
            "llm_skip_low": 0.15,
            "clearly_diverse": 0.25
        },
        "max_tokens": {
            "alignment": 1500,
            "aspect_overlap": 800,
            "aspect_suitability": 500,
            "turn_diversity": 500,
            "aspect_extraction": 800,
            "aspect_extraction_batch": 2000,
            "document_alignment": 1000,
            "semantic_facts": 500,
            "fact_verification": 1000,
            "subquestion": 500,
            "reasoning": 1000,
            "subquestion_reasoning": 800,
            "answerability": 800,
            "document_scoring": 2000,
            "query": 400,
            "answer": 1000,
            "aspect_overlap_check": 500,
            "semantic_similarity": 200
        }
    },
    
    "processing": {
        "max_decomposition_attempts": 3,
        "alignment_min_coverage": 0.50,
        "alignment_fallback_coverage": 0.15,
        "batch_size_default": 5,
        "max_workers_default": 5
    }
}


# ============================================================
# CONFIGURATION VALIDATOR
# ============================================================

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration on startup to catch errors early."""
    
    assert config["aspects"]["min_count"] > 0, "min_count must be positive"
    assert config["aspects"]["max_count"]["batch"] >= config["aspects"]["min_count"], \
        "max_count batch must be >= min_count"
    assert config["aspects"]["max_count"]["single"] >= config["aspects"]["min_count"], \
        "max_count single must be >= min_count"
    assert 0 <= config["aspects"]["document_coverage_min"] <= 1.0, \
        "document_coverage_min must be between 0 and 1"
    assert 0 <= config["aspects"]["overlap_confidence_threshold"] <= 1.0, \
        "overlap_confidence_threshold must be between 0 and 1"
    
    assert config["turns"]["min_count"] > 0, "turn min_count must be positive"
    assert 0 <= config["turns"]["min_answerability"] <= 1.0, \
        "min_answerability must be between 0 and 1"
    assert 0 <= config["turns"]["diversity"]["max_similarity"] <= 1.0, \
        "diversity max_similarity must be between 0 and 1"
    assert config["turns"]["diversity"]["min_new_words"] >= 0, \
        "diversity min_new_words must be non-negative"
    
    assert config["documents"]["scoring_threshold"] >= 0, \
        "scoring_threshold must be non-negative"
    assert config["documents"]["per_turn_default"] > 0, \
        "per_turn_default must be positive"
    
    for temp_key, temp_val in config["llm"]["temperature"].items():
        assert 0 <= temp_val <= 2.0, f"temperature {temp_key} must be between 0 and 2"
    
    for token_key, token_val in config["llm"]["max_tokens"].items():
        assert token_val > 0, f"max_tokens {token_key} must be positive"
    
    assert config["processing"]["max_decomposition_attempts"] > 0, \
        "max_decomposition_attempts must be positive"
    assert 0 <= config["processing"]["alignment_min_coverage"] <= 1.0, \
        "alignment_min_coverage must be between 0 and 1"
    
    assert config["subquestions"]["max_generation_retries"] > 0, \
        "max_generation_retries must be positive"
    
    logger.info("✓ Configuration validated successfully")


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(output_dir: str = ".") -> logging.Logger:
    """Setup logging to file and console."""
    log_file = os.path.join(output_dir, f"conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger("bright_unified")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()


# ============================================================
# AZURE OPENAI CLIENT
# ============================================================

class AzureOpenAIClient:
    """Azure OpenAI wrapper with error handling, thread safety, and retry logic."""
    
    def __init__(self):
        
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")
        
        if not all([self.endpoint, self.api_key, self.api_version, self.deployment_name]):
            raise RuntimeError(
                "Missing Azure OpenAI environment variables. "
                "Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_API_VERSION, and AZURE_DEPLOYMENT_NAME."
            )
        
        self._local = threading.local()
    
    def _get_client(self):
        """Get or create thread-local client."""
        if not hasattr(self._local, 'client'):
            self._local.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
            )
        return self._local.client
    
    def call_gpt_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Call Azure OpenAI and return parsed JSON with 3-attempt retry."""
        error_msg = None
        
        for attempt in range(3):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                parsed = json.loads(content)
                return parsed, None
            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing error: {str(e)}"
                logger.warning(f"Attempt {attempt + 1}/3 failed: {error_msg}")
            except Exception as e:
                error_msg = f"LLM call error: {str(e)}"
                logger.warning(f"Attempt {attempt + 1}/3 failed: {error_msg}")
            
            if attempt < 2:
                time.sleep(1)
        
        return None, error_msg
    
    def call_gpt_text(
        self,
        user_prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Tuple[Optional[str], Optional[str]]:
        """Call GPT and return free-form text response with retry logic."""
        error_msg = None
        
        for attempt in range(3):
            try:
                client = self._get_client()
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                
                response = client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                return content, None
            except Exception as e:
                error_msg = f"LLM call error: {str(e)}"
                logger.warning(f"Attempt {attempt + 1}/3 failed: {error_msg}")
            
            if attempt < 2:
                time.sleep(1)
        
        return None, error_msg


azure_client = None

def get_azure_client() -> AzureOpenAIClient:
    """Get or initialize Azure OpenAI client."""
    global azure_client
    if azure_client is None:
        azure_client = AzureOpenAIClient()
    return azure_client


# ============================================================
# JSON RESPONSE VALIDATION
# ============================================================

def validate_json_response(
    result: Optional[Dict[str, Any]],
    required_keys: List[str],
    type_checks: Optional[Dict[str, type]] = None,
    min_lengths: Optional[Dict[str, int]] = None
) -> bool:
    """Validate LLM JSON response structure."""
    if result is None:
        return False
    
    for key in required_keys:
        if key not in result:
            return False
    
    if type_checks:
        for key, expected_type in type_checks.items():
            if key in result and not isinstance(result[key], expected_type):
                return False
    
    if min_lengths:
        for key, min_len in min_lengths.items():
            if key in result:
                value = result[key]
                if isinstance(value, str) and len(value.strip()) < min_len:
                    return False
    
    return True


# ============================================================
# HTML CLEANING UTILITY
# ============================================================

def clean_html(html_text: str) -> str:
    """Remove HTML tags and clean up text."""
    if not html_text:
        return ""
    clean = re.sub(r'<[^>]+>', ' ', html_text)
    clean = clean.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    clean = clean.replace('&#39;', "'").replace('&quot;', '"').replace('&nbsp;', ' ')
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


# ============================================================
# LOAD BRIGHT DATASET
# ============================================================

def load_bright_data(task_split: str) -> Tuple[Any, Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Load BRIGHT dataset with GPT-4 reasoning AND human reasoning.
    
    Returns:
        examples_ds: Dataset with examples
        documents_dict: {doc_id: content}
        gpt4_reasoning_dict: {example_id: gpt4_long_reasoning}
        human_reasoning_dict: {example_id: human_short_reasoning}
    """
    logger.info(f"Loading BRIGHT dataset: {task_split}")
    
    try:
        examples_ds = load_dataset("xlangai/BRIGHT", "examples", split=task_split)
        docs_ds = load_dataset("xlangai/BRIGHT", "documents", split=task_split)
        gpt4_reasons_ds = load_dataset("xlangai/BRIGHT", "gpt4_reason", split=task_split)
        
        documents_dict = {row["id"]: row["content"] for row in docs_ds}
        
        gpt4_reasoning_dict = {row["id"]: row["query"] for row in gpt4_reasons_ds}
        
        human_reasoning_dict = {ex["id"]: ex.get("reasoning", "") for ex in examples_ds}
        
        logger.info(f"Loaded {len(examples_ds)} examples, {len(documents_dict)} documents")
        logger.info(f"GPT-4 reasoning: {len(gpt4_reasoning_dict)} entries")
        logger.info(f"Human reasoning: {len(human_reasoning_dict)} entries")
        return examples_ds, documents_dict, gpt4_reasoning_dict, human_reasoning_dict
    
    except Exception as e:
        logger.error(f"Failed to load BRIGHT data: {e}")
        raise


# ============================================================
# GPT-4 REASONING EXTRACTION
# ============================================================

def extract_gpt4_reasoning_sections(gpt4_reasoning: str) -> Dict[str, str]:
    """Extract structured sections from GPT-4 reasoning."""
    sections = {
        "essential_problem": "",
        "relevant_information": "",
        "reasoning_steps": "",
        "answer_framework": ""
    }
    
    parts = re.split(r'##\s+', gpt4_reasoning)
    
    markers = [
        ("essential problem", "essential_problem"),
        ("relevant information", "relevant_information"),
        ("step-by-step", "reasoning_steps"),
        ("drafting an answer", "answer_framework"),
        ("conclusion", "answer_framework"),
    ]
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        for marker_text, key in markers:
            if part.lower().startswith(marker_text):
                content = part[len(marker_text):].strip()
                if content.startswith(":"):
                    content = content[1:].strip()
                sections[key] = content
                break
    
    if not any(sections.values()):
        sections["essential_problem"] = gpt4_reasoning[:500]
    
    return sections


# ============================================================
# CHECKPOINT MANAGEMENT
# ============================================================

def load_processed_ids(output_path: str) -> set:
    """Load IDs that have already been processed."""
    processed_ids = set()
    
    if not os.path.exists(output_path):
        logger.info(f"No existing output file found. Starting fresh.")
        return processed_ids
    
    try:
        with open(output_path, "r", encoding="utf-8") as fin:
            for line in fin:
                try:
                    data = json.loads(line.strip())
                    if "id" in data:
                        processed_ids.add(data["id"])
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(processed_ids)} processed IDs from checkpoint")
        return processed_ids
    
    except Exception as e:
        logger.warning(f"Error loading checkpoint: {e}")
        return set()



# ============================================================
# CHECKPOINT FILE WRITER WITH THREAD SAFETY
# ============================================================

class CheckpointWriter:
    """Thread-safe writer for immediate checkpoint saving."""
    
    def __init__(self, output_dir: str, task_split: str):
        self.output_dir = output_dir
        self.task_split = task_split
        
        self.benchmark_path = os.path.join(output_dir, f"{task_split}_benchmark.jsonl")
        self.analysis_path = os.path.join(output_dir, f"{task_split}_analysis.jsonl")
        self.debug_path = os.path.join(output_dir, f"{task_split}_debug.jsonl")
        self.docs_path = os.path.join(output_dir, f"{task_split}_documents.jsonl")
        
        self.write_lock = threading.Lock()
        self.docs_dict = {}
        self.docs_lock = threading.Lock()
        
        logger.info(f"✓ Checkpoint writer initialized for {task_split}")
    
    def write_record(self, result_full: Dict[str, Any]) -> bool:
        """Write a single record to all output files immediately."""
        try:
            with self.write_lock:
                # Write benchmark entry
                benchmark_entry = create_benchmark_entry(result_full)
                with open(self.benchmark_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(benchmark_entry, ensure_ascii=False) + "\n")
                    f.flush()
                
                # Write analysis entry
                analysis_entry = create_analysis_entry(result_full)
                with open(self.analysis_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(analysis_entry, ensure_ascii=False) + "\n")
                    f.flush()
                
                # Write debug entry
                with open(self.debug_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result_full, ensure_ascii=False) + "\n")
                    f.flush()
                
                # Extract and save documents incrementally
                self.extract_and_save_documents(result_full)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to write checkpoint: {e}")
            return False
    
    def extract_and_save_documents(self, result_full: Dict[str, Any]) -> None:
        """Extract documents from entry and save incrementally."""
        with self.docs_lock:
            new_docs = {}
            for turn in result_full.get("turns", []):
                for doc in turn.get("supporting_documents", []):
                    doc_id = doc["doc_id"]
                    if doc_id not in self.docs_dict:
                        self.docs_dict[doc_id] = doc["content"]
                        new_docs[doc_id] = doc["content"]
            
            # Append new documents immediately
            if new_docs:
                with open(self.docs_path, "a", encoding="utf-8") as f:
                    for doc_id, content in sorted(new_docs.items()):
                        f.write(json.dumps({
                            "doc_id": doc_id,
                            "content": content
                        }, ensure_ascii=False) + "\n")
                    f.flush()
    
    def get_unique_doc_count(self) -> int:
        """Get count of unique documents."""
        with self.docs_lock:
            return len(self.docs_dict)

# ============================================================
# DOCUMENT-GOLD ALIGNMENT VALIDATION
# ============================================================

def validate_gold_document_alignment_llm(
    gold_answer: str,
    documents: Dict[str, str],
    client: AzureOpenAIClient
) -> Tuple[bool, float, List[str]]:
    """
    Use LLM to check if documents can support the gold answer.
    
    Returns:
        - is_sufficient: bool (True if coverage >= threshold)
        - coverage_percentage: float (0.0 - 1.0)
        - unsupported_claims: List[str]
    """
    doc_summary = "\n---\n".join(
        f"[Doc {i+1} ({doc_id})]: {doc}"
        for i, (doc_id, doc) in enumerate(documents.items())
    )
    
    min_coverage = CONFIG["processing"]["alignment_min_coverage"]
    
    prompt = f"""You are evaluating whether documents can support an answer.

GOLD ANSWER:
{gold_answer}

DOCUMENTS:
{doc_summary}

IMPORTANT: Be GENEROUS in your assessment:
- Documents don't need ALL details
- Core facts are enough
- Related information counts as support

TASK:
1. Identify KEY CLAIMS (focus on main ideas, not minor details, max 3-5 claims)
2. Check if documents contain supporting information
3. Return coverage percentage

Return JSON:
{{
    "key_claims": ["main claim 1", "main claim 2", ...],
    "supported_claims": ["claim1", ...],
    "unsupported_claims": ["claim2", ...],
    "coverage_percentage": 0.0-1.0,
    "is_sufficient": boolean
}}

Set is_sufficient to true if coverage_percentage >= {min_coverage}"""
    
    result, error = client.call_gpt_json(
        system_prompt="You evaluate document coverage for answers. Be thorough and GENEROUS.",
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"]["alignment"],
        max_tokens=CONFIG["llm"]["max_tokens"]["alignment"]
    )
    
    if error or not result:
        logger.warning(f"Document-gold alignment check failed: {error}")
        return True, 0.5, []
    
    is_sufficient = result.get("is_sufficient", True)
    coverage = result.get("coverage_percentage", 0.5)
    unsupported = result.get("unsupported_claims", [])
    
    return is_sufficient, coverage, unsupported


# ============================================================
# SUB-QUESTION SUITABILITY FILTER
# ============================================================

def should_generate_subquestion(
    aspect: Dict[str, Any],
    client: AzureOpenAIClient
) -> Tuple[bool, str, str]:
    """
    LLM determines if aspect should become a sub-question.
    
    Returns:
        - should_generate: bool
        - reason: str
        - category: "substantive" | "meta" | "insufficient"
    """
    prompt = f"""Should this aspect become a conversation sub-question?

ASPECT: {aspect.get('aspect_name', 'unnamed')}
TYPE: {aspect.get('aspect_type', 'unknown')}
EXCERPT: {aspect.get('excerpt', '')}

RULES:
- Substantive (facts, mechanisms, examples) → YES
- Meta-commentary (disclaimers, "consult professional") → NO
- Very short excerpts with clear facts → YES
- Meaningless or purely structural → NO

Return JSON:
{{
    "should_generate": boolean,
    "reason": "explanation",
    "aspect_category": "substantive" | "meta" | "insufficient"
}}"""
    
    result, error = client.call_gpt_json(
        system_prompt="You classify aspects for conversation generation.",
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"]["aspect_suitability"],
        max_tokens=CONFIG["llm"]["max_tokens"]["aspect_suitability"]
    )
    
    if error or not result:
        return True, "Validation check failed", "unknown"
    
    should_gen = result.get("should_generate", True)
    reason = result.get("reason", "")
    category = result.get("aspect_category", "unknown")
    
    return should_gen, reason, category


# ============================================================
# BATCH SUITABILITY CHECK (MODIFIED)
# ============================================================

def batch_check_suitability(
    aspects: List[Dict[str, Any]],
    client: AzureOpenAIClient
) -> List[Tuple[bool, str, str]]:
    """
    Check suitability of multiple aspects in ONE LLM call.
    Returns: List of (should_generate, reason, category) tuples
    """
    if not aspects:
        return []
    
    if len(aspects) <= 2:
        results = []
        for aspect in aspects:
            should_gen, reason, category = should_generate_subquestion(aspect, client)
            results.append((should_gen, reason, category))
        return results
    
    aspects_text = "\n\n".join([
        f"[INDEX {i}] ASPECT:\n- Name: {a.get('aspect_name', 'unnamed')}\n- Type: {a.get('aspect_type', 'unknown')}\n- Excerpt: {a.get('excerpt', '')}"
        for i, a in enumerate(aspects)
    ])
    
    num_aspects = len(aspects)
    
    prompt = f"""Evaluate which of the {num_aspects} aspects should become conversation sub-questions.

{aspects_text}

CLASSIFICATION RULES:
1. "substantive" → YES: Contains facts, mechanisms, explanations, examples, or concrete information.
   * NOTE: Short excerpts ARE ACCEPTABLE if they contain a clear fact.
2. "meta" → NO: Contains disclaimers, advice to "consult professional", caveats, or meta-commentary.
3. "insufficient" → NO: Content is meaningless or purely structural (e.g., "Here is a list:").

TASK: Evaluate ALL {num_aspects} aspects (indices 0 to {num_aspects - 1}).

Return JSON:
{{
    "evaluations": [
        {{
            "aspect_index": 0,
            "should_generate": true,
            "reason": "Contains concrete fact about dimensions",
            "category": "substantive"
        }}
    ]
}}

IMPORTANT: 
- Use 0-based indexing (0, 1, 2, ...)
- Include ALL {num_aspects} aspects.
- Accept SHORT excerpts if they are factual.
"""
    
    tokens_needed = min(60 * num_aspects + 100, 1500)
    
    result, error = client.call_gpt_json(
        system_prompt="You classify text aspects. Accept granular details even if short.",
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"]["aspect_suitability"],
        max_tokens=tokens_needed
    )
    
    if error or not result:
        logger.warning(f"Batch suitability check failed: {error}")
        return [(True, "Batch check failed", "unknown") for _ in aspects]
    
    evaluations = result.get("evaluations", [])
    
    results = []
    eval_dict = {}
    
    for e in evaluations:
        if "aspect_index" in e:
            idx = e["aspect_index"]
            if idx >= num_aspects and idx - 1 < num_aspects:
                idx = idx - 1
                logger.warning(f"Adjusted 1-based index {e['aspect_index']} to 0-based {idx}")
            eval_dict[idx] = e
    
    for i in range(num_aspects):
        if i in eval_dict:
            e = eval_dict[i]
            category = e.get("category", "unknown").lower()
            if category not in ["substantive", "meta", "insufficient"]:
                category = "unknown"
            
            results.append((
                e.get("should_generate", True),
                e.get("reason", ""),
                category
            ))
        else:
            logger.warning(f"Aspect index {i} not evaluated by LLM, assuming suitable")
            results.append((True, "Not evaluated by LLM", "unknown"))
    
    logger.info(f"Batch suitability: {sum(1 for r in results if r[0])}/{num_aspects} suitable")
    
    return results


# ============================================================
# LLM-BASED TURN DIVERSITY VALIDATION (MODIFIED)
# ============================================================

def validate_turn_diversity_llm(
    new_turn: Dict[str, Any],
    previous_turns: List[Dict[str, Any]],
    client: AzureOpenAIClient
) -> Tuple[bool, str]:
    """LLM validates that new turn adds value (distinct info OR deepening)."""
    if not previous_turns:
        return True, "First turn"
    
    # Get last 3 turns for context (not all history)
    recent_turns = previous_turns[-3:] if len(previous_turns) > 3 else previous_turns
    
    prev_content = "\n".join(
        f"Turn {i+1} Answer: {t.get('answer', '')}"
        for i, t in enumerate(recent_turns)
    )
    
    new_answer = new_turn.get('answer', '')
    
    prompt = f"""Does this new answer add value to the conversation?

RECENT CONVERSATION:
{prev_content}

NEW ANSWER:
{new_answer}

EVALUATION CRITERIA:

✓ ACCEPT if answer does ANY of these:
1. Introduces NEW factual information not previously stated
2. Drills deeper into a specific detail (e.g., mechanism → sub-mechanism)
3. Provides a concrete example of something described generally before
4. Explains a consequence/implication not yet discussed
5. Answers a different aspect of the same topic

✗ REJECT only if answer does BOTH:
1. Contains the SAME factual claims as previous turns (not just same topic)
2. Uses similar phrasing/wording to express those facts

IMPORTANT DISTINCTIONS:

Same TOPIC but different ANGLE = ACCEPT
Example: "What nasal cycle is" vs "How nervous system controls it" → ACCEPT

Same FACTS with different WORDS = REJECT
Example: Turn 1: "receptors are broadly tuned to detect chemicals"
         New: "receptors can sense a wide range of chemical molecules" → REJECT

General → Specific = ACCEPT
Example: Turn 1: "Insects are drawn to light"
         New: "Their photoreceptors detect specific wavelengths" → ACCEPT

Return JSON:
{{
    "adds_value": boolean,
    "value_type": "new_facts" | "deepening" | "example" | "implication" | "different_angle" | "repetitive",
    "reason": "brief explanation"
}}"""
    
    result, error = client.call_gpt_json(
        system_prompt="You evaluate if a conversational answer adds value. Accept answers that deepen understanding or cover different angles. Reject only pure repetition of the same facts.",
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"]["turn_diversity"],
        max_tokens=CONFIG["llm"]["max_tokens"]["turn_diversity"]
    )
    
    if error or not result:
        return True, "Validation check failed"
    
    adds_value = result.get("adds_value", True)
    value_type = result.get("value_type", "")
    reason = result.get("reason", "")
    
    if not adds_value:
        return False, f"{value_type}: {reason}"
    
    return True, f"{value_type}: {reason}"

# ============================================================
# EXTRACT MULTIPLE ASPECTS (MODIFIED FOR GRANULARITY)
# ============================================================

def extract_multiple_aspects_focused(
    query: str,
    reasoning: str,
    gold_answer: str,
    exclude_excerpts: List[str],
    existing_aspect_names: List[str],
    client: AzureOpenAIClient,
    num_aspects: int
) -> List[Dict[str, Any]]:
    """Extract MULTIPLE distinct, granular aspects from gold answer for conversation generation."""
    
    if exclude_excerpts:
        exclusion_text = "\n".join([f"{i+1}. \"{exc[:100]}...\"" for i, exc in enumerate(exclude_excerpts)])
    else:
        exclusion_text = "None"
    
    if existing_aspect_names:
        existing_text = "\n".join([f"- {name}" for name in existing_aspect_names])
    else:
        existing_text = "None"
    
    prompt = f"""Extract {num_aspects} distinct, GRANULAR aspects from the gold answer.

QUERY: {query}

REASONING (why this answer is relevant):
{reasoning}

GOLD ANSWER:
{gold_answer}

ALREADY EXTRACTED ASPECTS (do NOT duplicate these):
{existing_text}

ALREADY USED TEXT (use DIFFERENT portions):
{exclusion_text}

=== WHAT IS A GRANULAR ASPECT? ===
An aspect does NOT need to be a broad topic. It can be a specific detail, a single step in a process, or a distinct implication.
- Broad (AVOID): "How Photosynthesis Works" (Too big, covers everything)
- Granular 1 (GOOD): "Role of Chlorophyll in Light Absorption"
- Granular 2 (GOOD): "The Calvin Cycle's Carbon Fixation Step"

=== ASPECT REQUIREMENTS ===
1. SPECIFICITY: Drill down into details. A single sentence with a verified fact can be an aspect.
2. VERBATIM EXCERPT: Copy exact text from gold answer.
3. SUBSTANTIVE: Must contain facts, mechanisms, or examples.
4. DISTINCT ANGLE: If a topic is already covered, look for a specific *implication*, *limitation*, or *counter-example* not yet discussed.

=== ASPECT TYPES ===
- "detail": A specific fact or component of a larger system
- "step": A single stage in a process or mechanism
- "implication": A consequence or result of a fact
- "distinction": A subtle difference between two related concepts
- "definition": Explains what something IS
- "mechanism": Explains HOW something works
- "example": Provides concrete cases
- "comparison": Contrasts two or more things
- "history": Covers evolution, origins, or timeline
- "application": Explains when/why to use something

CRITICAL: Balance Breadth and Depth
- It is OK to have multiple aspects on the same general topic IF they cover different specific details.
- Example GOOD: 
  1. "General definition of FPV motors"
  2. "Specific role of ball bearings in FPV motors"
  3. "Difference between stator sizes"
  (These are all "FPV motors" but distinct granular details)

=== OVERLAP DETECTION ===
- "Photosynthesis Overview" vs "How Plants Eat" -> OVERLAP (Same concept)
- "Light Dependent Reactions" vs "Calvin Cycle" -> DISTINCT (Different steps of same process) -> KEEP BOTH

Return JSON:
{{
    "aspects": [
        {{
            "aspect_name": "Specific Name (3-6 words)",
            "aspect_type": "detail|step|implication|distinction|definition|mechanism|example|comparison|history|application",
            "excerpt": "Exact verbatim text from gold answer. Must be 1+ sentences.",
            "distinct_from_existing": "How this specific detail differs from existing aspects"
        }}
    ],
    "extraction_notes": "Notes on finding distinct details"
}}

IMPORTANT: Dig deep into the text. If the main concepts are taken, look for specific constraints, historical exceptions, or minor mechanical details. Do not stop at broad topics."""

    result, error = client.call_gpt_json(
        system_prompt="You extract granular topical aspects from educational text. Focus on specific details, steps, and implications. Separate broad concepts into their component parts.",
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"].get("aspect_extraction", 0.3),
        max_tokens=CONFIG["llm"]["max_tokens"].get("aspect_extraction_batch", 1500)
    )
    
    if error or not result:
        logger.warning(f"Aspect extraction failed: {error}")
        return []
    
    aspects = result.get("aspects", [])
    extraction_notes = result.get("extraction_notes", "")
    
    if extraction_notes:
        logger.info(f"Extraction notes: {extraction_notes}")
    
    valid_aspects = []
    for aspect in aspects:
        if not all(k in aspect for k in ["aspect_name", "excerpt"]):
            logger.warning(f"Aspect missing required fields: {aspect.get('aspect_name', 'unnamed')}")
            continue
        
        excerpt = aspect.get("excerpt", "")
        word_count = len(excerpt.split())
        
        if word_count < 10:
            logger.warning(f"Aspect '{aspect['aspect_name']}' excerpt too short ({word_count} words)")
        
        name = aspect.get("aspect_name", "")
        if name.endswith("?") or name.lower().startswith(("what ", "how ", "why ", "when ", "where ")):
            logger.warning(f"Aspect name is a question, skipping: {name}")
            continue
        
        aspect_type = aspect.get("aspect_type", "factual").lower()
        valid_types = {"definition", "mechanism", "example", "comparison", "history", "application", "detail", "step", "implication", "distinction"}
        if aspect_type not in valid_types:
            aspect_type = "detail"
        
        valid_aspects.append({
            "aspect_name": name.strip(),
            "aspect_type": aspect_type,
            "excerpt": excerpt.strip(),
            "distinct_from_existing": aspect.get("distinct_from_existing", "")
        })
    
    logger.info(f"Extracted {len(valid_aspects)}/{len(aspects)} valid aspects")
    
    return valid_aspects


# ============================================================
# CHECK ASPECT DOCUMENT ALIGNMENT
# ============================================================

def check_aspect_document_alignment(
    aspect_excerpt: str,
    aspect_name: str,
    documents: Dict[str, str],
    client: AzureOpenAIClient
) -> Tuple[float, List[str], str, bool, str]:
    """
    Returns:
        - coverage_score, supporting_evidence, missing_info (existing)
        - name_grounded: bool (is aspect_name grounded in docs?)
        - suggested_name: str (document-grounded alternative name)
    """
    docs_text = "\n---\n".join([
        f"[{doc_id}]: {content}"
        for doc_id, content in documents.items()
    ])
    
    prompt = f"""Evaluate this aspect against the documents.

ASPECT NAME: {aspect_name}
ASPECT CONTENT: {aspect_excerpt}

DOCUMENTS:
{docs_text}

TASK:
1. Check if ASPECT CONTENT is supported by documents
2. Check if ASPECT NAME uses only entities/concepts FROM the documents
   - If aspect_name mentions specific people, events, treaties NOT in documents → name_grounded = false
3. If name is not grounded, suggest an alternative name using ONLY document terminology

Return JSON:
{{
    "coverage_score": 0.0-1.0,
    "key_facts_in_aspect": ["fact1", "fact2"],
    "facts_found": ["fact1"],
    "facts_missing": ["fact2"],
    "supporting_doc_ids": ["doc_0"],
    "missing_information": "what's not in docs",
    "name_grounded": boolean,
    "name_issues": "what entities in name are not in docs",
    "suggested_name": "document-grounded alternative name"
}}"""
    
    result, error = client.call_gpt_json(
        system_prompt="You evaluate aspect alignment with documents.",
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"]["alignment"],
        max_tokens=CONFIG["llm"]["max_tokens"]["document_alignment"]
    )
    
    if error or not result:
        return 0.5, [], "Check failed", True, aspect_name
    
    coverage = result.get("coverage_score", 0.0)
    supporting_docs = result.get("supporting_doc_ids", [])
    missing_info = result.get("missing_information", "Unknown")
    name_grounded = result.get("name_grounded", True)
    suggested_name = result.get("suggested_name", aspect_name)
    
    return coverage, supporting_docs, missing_info, name_grounded, suggested_name


# ============================================================
# SEMANTIC FACT SIMILARITY
# ============================================================

def compute_semantic_fact_similarity(
    facts1: List[str],
    facts2: List[str],
    client: Optional[AzureOpenAIClient] = None
) -> float:
    """Compute semantic similarity between two sets of facts using LLM."""
    
    if not facts1 or not facts2:
        return 0.0
    
    if not client:
        client = get_azure_client()
    
    facts1_text = "\n".join([f"- {f}" for f in facts1[:8]])
    facts2_text = "\n".join([f"- {f}" for f in facts2[:8]])
    
    prompt = f"""Rate the semantic similarity between these two sets of facts.

SET A (Facts):
{facts1_text}

SET B (Facts):
{facts2_text}

SCORING GUIDE:
- 0.0-0.2: Completely unrelated facts about different topics
- 0.3-0.4: Same general domain but different specific facts
- 0.5-0.6: Some overlapping concepts but distinct information
- 0.7-0.8: Mostly the same information with minor differences
- 0.9-1.0: Nearly identical facts, just different wording

EXAMPLES:
Set A: ["Nations share cultural identity", "Language unifies nations"]
Set B: ["Cultural identity defines nations", "Common language in nations"]
→ similarity: 0.85 (same concepts, different wording)

Set A: ["Nations share cultural identity", "Language unifies nations"]
Set B: ["States have defined borders", "Sovereignty defines states"]
→ similarity: 0.25 (related domain but different concepts)

Set A: ["Definition of parliamentary system", "Role of prime minister"]
Set B: ["Historical evolution of parliaments", "Famous parliamentary debates"]
→ similarity: 0.40 (same topic area, different aspects)

Return JSON:
{{
    "similarity": 0.0-1.0,
    "reason": "brief explanation"
}}"""

    result, error = client.call_gpt_json(
        system_prompt="You assess semantic similarity between fact sets. Focus on whether facts convey the SAME information, not just related topics.",
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"].get("semantic_similarity", 0.2),
        max_tokens=CONFIG["llm"]["max_tokens"].get("semantic_similarity", 200)
    )
    
    if error or not result:
        logger.warning(f"Semantic similarity check failed: {error}")
        return 0.5
    
    return result.get("similarity", 0.5)


# ============================================================
# CHECK OVERLAP WITH EXISTING ASPECTS (MODIFIED)
# ============================================================

def check_aspect_overlap_with_existing(
    new_aspect: Dict[str, Any],
    existing_aspects: List[Dict[str, Any]],
    client: AzureOpenAIClient
) -> Tuple[bool, str]:
    """Check if new aspect's CONTENT overlaps with existing aspects."""
    
    if not existing_aspects:
        return False, "No existing aspects"
    
    existing_text = "\n".join([
        f"[{i+1}] {a['aspect_name']} ({a.get('aspect_type', 'unknown')})\n    Content: {a.get('excerpt', '')}"
        for i, a in enumerate(existing_aspects)
    ])
    
    new_excerpt = new_aspect.get('excerpt', '')
    new_type = new_aspect.get('aspect_type', 'unknown')
    
    prompt = f"""Does the NEW ASPECT cover content already covered by EXISTING ASPECTS?

NEW ASPECT: {new_aspect['aspect_name']} ({new_type})
CONTENT: {new_excerpt}

EXISTING ASPECTS:
{existing_text}

WHAT COUNTS AS OVERLAP?
Two aspects OVERLAP if they:
- Make the EXACT SAME factual claims.
- Explain the EXACT SAME step of a process.
- Are merely rephrasing the same information.

WHAT IS NOT OVERLAP? (DISTINCT CONTENT):
- Different steps of the *same* mechanism (e.g., "Step 1: Input" vs "Step 2: Processing").
- Different examples of the *same* concept.
- Specific details vs General definitions (e.g., "General Engine" vs "Fuel Injection System").
- Implications vs Mechanisms.
- "What it is" vs "How it works" vs "Why it matters".

DECISION: Does the new aspect contain specific details, steps, or implications NOT in the existing aspects?

Return JSON:
{{
    "has_overlap": boolean,
    "overlaps_with": "name of overlapping aspect" or null,
    "overlap_type": "same_claims" | "same_examples" | "same_step" | "no_overlap",
    "reasoning": "one sentence explanation"
}}"""

    result, error = client.call_gpt_json(
        system_prompt="You detect content overlap. Allow granular details and specific steps to be distinct from general concepts. Only flag exact factual repetition.",
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"].get("aspect_overlap_check", 0.2),
        max_tokens=CONFIG["llm"]["max_tokens"].get("aspect_overlap_check", 300)
    )
    
    if error or not result:
        logger.warning(f"Overlap check failed: {error}")
        return False, "Check failed, assuming no overlap"
    
    has_overlap = result.get("has_overlap", False)
    reasoning = result.get("reasoning", "")
    overlaps_with = result.get("overlaps_with", "")
    
    if has_overlap:
        return True, f"Overlaps with '{overlaps_with}': {reasoning}"
    
    return False, reasoning


# ============================================================
# SEMANTIC FACTS EXTRACTION
# ============================================================

SYSTEM_PROMPT_SEMANTIC_FACTS = """You extract key facts from text as SHORT bullet points (5-10 words each)."""

USER_PROMPT_SEMANTIC_FACTS = """Extract at least 3 key facts as SHORT bullet points.

ASPECT EXCERPT:
{aspect_excerpt}

INSTRUCTIONS:
1. Extract SPECIFIC, VERIFIABLE facts (not opinions or vague statements)
2. Each fact: 5-10 words maximum
3. Include actionable details (what, how, why)
4. Do NOT copy full sentences

GOOD: "sealed bearing design prevents contamination"
GOOD: "pre-lubricated bearings require no maintenance"
BAD: "The bearings are sealed." (too generic)
BAD: "Motors are important." (too vague)

Return JSON:
{{
  "semantic_facts": [
    "fact 1 (5-10 words)",
    "fact 2 (5-10 words)",
    "fact 3 (5-10 words)"
  ],
  "total_facts": integer
}}"""


def extract_semantic_facts_from_excerpt(aspect_excerpt: str) -> Tuple[List[str], Dict[str, Any]]:
    """Extract semantic facts as bullet points from aspect excerpt."""
    client = get_azure_client()
    
    metadata = {"extraction_method": "llm"}
    
    prompt = USER_PROMPT_SEMANTIC_FACTS.format(aspect_excerpt=aspect_excerpt)
    
    result, error = client.call_gpt_json(
        system_prompt=SYSTEM_PROMPT_SEMANTIC_FACTS,
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"]["semantic_facts"],
        max_tokens=CONFIG["llm"]["max_tokens"]["semantic_facts"]
    )
    
    if error:
        sentences = [s.strip() for s in aspect_excerpt.split('.') if s.strip()]
        facts = [s[:50] + "..." if len(s) > 50 else s for s in sentences]
        metadata["fallback"] = True
        metadata["error"] = error
        return facts, metadata
    
    if not validate_json_response(result, ["semantic_facts"], {"semantic_facts": list}):
        sentences = [s.strip() for s in aspect_excerpt.split('.') if s.strip()]
        facts = [s[:50] + "..." if len(s) > 50 else s for s in sentences]
        metadata["fallback"] = True
        return facts, metadata
    
    facts = result.get("semantic_facts", [])
    
    if not facts or len(facts) < 2:
        sentences = [s.strip() for s in aspect_excerpt.split('.') if s.strip()]
        facts = [s[:50] + "..." if len(s) > 50 else s for s in sentences]
        metadata["fallback"] = True
        return facts, metadata
    
    metadata["success"] = True
    metadata["total_facts"] = len(facts)
    return facts, metadata


# ============================================================
# VERIFY SEMANTIC FACTS AGAINST DOCUMENTS
# ============================================================

def verify_facts_against_documents(
    semantic_facts: List[str],
    documents: Dict[str, str],
    client: AzureOpenAIClient
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Verify which semantic facts are supported by documents.
    Returns: (supported_facts, metadata)
    """
    if not semantic_facts:
        return [], {"verification_method": "none", "all_facts_supported": True}
    
    docs_text = "\n".join([
        f"DOC {doc_id}:\n{content}"
        for doc_id, content in documents.items()
    ])
    
    facts_text = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(semantic_facts)])
    
    prompt = f"""Verify which facts are supported by the documents.

FACTS TO VERIFY:
{facts_text}

DOCUMENTS:
{docs_text}

TASK:
For each fact, determine if it's supported by the documents.

A fact is "supported" if:
1. Explicitly stated in documents, OR
2. Clearly implied by combining document information, OR  
3. A reasonable paraphrase using different terminology

Only reject if fact introduces NEW information not present in documents.

Be LENIENT with paraphrases and terminology variations.

Return JSON:
{{
    "fact_verification": [
        {{
            "fact_number": 1,
            "fact_text": "the fact",
            "is_supported": boolean,
            "supporting_doc_ids": ["doc_0"],
            "reason": "why supported/not supported"
        }}
    ],
    "summary": {{
        "total_facts": integer,
        "supported_count": integer,
        "unsupported_count": integer
    }}
}}"""
    
    result, error = client.call_gpt_json(
        system_prompt="You verify fact support in documents. Be accurate but practical.",
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"]["fact_verification"],
        max_tokens=CONFIG["llm"]["max_tokens"]["fact_verification"]
    )
    
    metadata = {"verification_method": "llm"}
    
    if error or not result:
        logger.warning(f"Fact verification failed: {error}")
        metadata["fallback"] = True
        metadata["error"] = error
        return semantic_facts, metadata
    
    if not validate_json_response(result, ["fact_verification"], {"fact_verification": list}):
        logger.warning("Fact verification response validation failed")
        metadata["fallback"] = True
        return semantic_facts, metadata
    
    verifications = result.get("fact_verification", [])
    summary = result.get("summary", {})
    
    supported_facts = []
    unsupported_facts = []
    
    for verification in verifications:
        is_supported = verification.get("is_supported", False)
        fact_text = verification.get("fact_text", "")
        
        if is_supported:
            supported_facts.append(fact_text)
        else:
            unsupported_facts.append(fact_text)
    
    metadata["success"] = True
    metadata["total_facts"] = len(semantic_facts)
    metadata["supported_count"] = len(supported_facts)
    metadata["unsupported_count"] = len(unsupported_facts)
    metadata["unsupported_facts"] = unsupported_facts
    metadata["summary"] = summary
    
    logger.info(f"Fact verification: {len(supported_facts)}/{len(semantic_facts)} facts supported")
    
    if not supported_facts:
        logger.warning("No facts were verified as supported - returning empty list")
        metadata["fallback_no_supported"] = True
        return [], metadata
    
    return supported_facts, metadata


# ============================================================
# COMBINED FACT EXTRACTION AND VERIFICATION (MODIFIED)
# ============================================================

SYSTEM_PROMPT_EXTRACT_VERIFY_FACTS = """You extract and verify facts from text against source documents.
For short excerpts, it is perfectly acceptable to extract just 1 single, accurate fact."""

USER_PROMPT_EXTRACT_VERIFY_FACTS = """Extract key facts from the EXCERPT, then verify each against the DOCUMENTS.

ASPECT EXCERPT:
{aspect_excerpt}

DOCUMENTS:
{documents}

TASK:
1. Extract **1 to 5** key facts from the excerpt (depending on length).
   - If the excerpt is a single detail, extract just **1 fact**.
   - Do NOT invent facts to fill a quota.
2. For each fact, check if it's supported by the documents.
3. Return ONLY facts that are supported.

A fact is "supported" if:
- Explicitly stated in documents, OR
- Clearly implied by combining document information.

Return JSON:
{{
    "extracted_facts": [
        {{
            "fact": "the extracted fact (5-15 words)",
            "is_supported": boolean,
            "supporting_doc_id": "doc_X" or null,
            "reason": "brief reason"
        }}
    ],
    "supported_facts": ["fact1", "fact2", ...],
    "unsupported_facts": ["fact3", ...],
    "summary": {{
        "total_extracted": integer,
        "supported_count": integer
    }}
}}"""


def extract_and_verify_facts(
    aspect_excerpt: str,
    documents: Dict[str, str],
    client: AzureOpenAIClient
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Extract facts from excerpt AND verify against documents in ONE call.
    Modified to support granular (single-fact) aspects.
    """
    docs_text = "\n---\n".join([
        f"[{doc_id}]: {content[:800]}" if len(content) > 800 else f"[{doc_id}]: {content}"
        for doc_id, content in documents.items()
    ])
    
    prompt = USER_PROMPT_EXTRACT_VERIFY_FACTS.format(
        aspect_excerpt=aspect_excerpt,
        documents=docs_text
    )
    
    result, error = client.call_gpt_json(
        system_prompt=SYSTEM_PROMPT_EXTRACT_VERIFY_FACTS,
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"]["semantic_facts"],
        max_tokens=CONFIG["llm"]["max_tokens"]["fact_verification"]
    )
    
    metadata = {"method": "combined_extract_verify"}
    
    if error or not result:
        sentences = [s.strip() for s in aspect_excerpt.split('.') if s.strip()]
        facts = [s[:100] for s in sentences[:3]]
        metadata["fallback"] = True
        metadata["error"] = error
        return facts, facts, metadata
    
    supported_facts = result.get("supported_facts", [])
    unsupported_facts = result.get("unsupported_facts", [])
    all_extracted = supported_facts + unsupported_facts
    summary = result.get("summary", {})
    
    metadata["success"] = True
    metadata["total_extracted"] = len(all_extracted)
    metadata["supported_count"] = len(supported_facts)
    metadata["unsupported_count"] = len(unsupported_facts)
    metadata["unsupported_facts"] = unsupported_facts
    metadata["summary"] = summary
    
    if not supported_facts and all_extracted:
        logger.info(f"Strict verification yielded 0 facts. Using {len(all_extracted)} extracted facts as fallback.")
        metadata["fallback_no_supported"] = True
        return all_extracted, all_extracted, metadata
    
    if not supported_facts and not all_extracted:
        return [], [], metadata

    return supported_facts, all_extracted, metadata


# ============================================================
# BATCH ASPECT EXTRACTION WITH DOCUMENT SUPPORT (MODIFIED)
# ============================================================

def identify_aspects_with_document_support_iterative_BATCH(
    query: str,
    reasoning: str,
    gold_answer: str,
    documents: Dict[str, str],
    client: AzureOpenAIClient,
    min_aspects: int,
    max_aspects: int,
    max_extraction_attempts: int
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """Extract and validate aspects in BATCHES without early stopping."""
    
    validated_aspects = []
    processed_excerpts = []
    extraction_attempts = 0
    
    diagnostic = {
        "has_issue": False,
        "issue_type": None,
        "aspects_validated": 0,
        "aspects_rejected": 0,
        "extraction_attempts": 0,
        "rejection_reasons": [],
        "fallback_accepted": False,
        "success": False
    }
    
    while len(validated_aspects) < max_aspects and extraction_attempts < max_extraction_attempts:
        extraction_attempts += 1
        diagnostic["extraction_attempts"] = extraction_attempts
        
        needed = max_aspects - len(validated_aspects)
        num_to_extract = min(needed + 1, CONFIG["batch"]["aspects_per_attempt"])
        
        aspects = extract_multiple_aspects_focused(
            query=query,
            reasoning=reasoning,
            gold_answer=gold_answer,
            exclude_excerpts=processed_excerpts,
            existing_aspect_names=[a.get("aspect_name") for a in validated_aspects],
            client=client,
            num_aspects=num_to_extract
        )
        
        if not aspects:
            logger.info(f"No more extractable aspects after {extraction_attempts} attempts")
            break
        
        logger.info(f"Extraction attempt {extraction_attempts}: Got {len(aspects)} candidate aspects")
        
        suitability_results = batch_check_suitability(aspects, client)
        
        for aspect_idx, aspect in enumerate(aspects):
            if len(validated_aspects) >= max_aspects:
                break
            
            processed_excerpts.append(aspect["excerpt"])
            
            is_suitable, suitability_reason, category = suitability_results[aspect_idx]
            
            if not is_suitable:
                logger.info(f"Aspect '{aspect['aspect_name']}' rejected: {suitability_reason}")
                diagnostic["aspects_rejected"] += 1
                diagnostic["rejection_reasons"].append(f"suitability: {category}")
                continue
            
            supported_facts, all_facts, facts_metadata = extract_and_verify_facts(
                aspect["excerpt"], documents, client
            )
            aspect["semantic_facts"] = supported_facts
            aspect["semantic_facts_all"] = all_facts
            aspect["facts_metadata"] = facts_metadata
            
            if not supported_facts:
                logger.info(f"Aspect '{aspect['aspect_name']}' rejected: no verified facts")
                diagnostic["aspects_rejected"] += 1
                diagnostic["rejection_reasons"].append("no_verified_facts")
                continue
            
            if validated_aspects:
                has_overlap, overlap_reason = check_aspect_overlap_with_existing(
                    aspect, validated_aspects, client
                )
                
                if has_overlap:
                    logger.info(f"Aspect '{aspect['aspect_name']}' rejected: {overlap_reason}")
                    diagnostic["aspects_rejected"] += 1
                    diagnostic["rejection_reasons"].append(f"overlap: {overlap_reason}")
                    continue
            
            aspect_coverage, supporting_evidence, missing_info, name_grounded, suggested_name = check_aspect_document_alignment(
                aspect["excerpt"], aspect["aspect_name"], documents, client
            )
            
            if not name_grounded:
                logger.info(f"Aspect name '{aspect['aspect_name']}' not grounded, using: '{suggested_name}'")
                aspect["aspect_name_original"] = aspect["aspect_name"]
                aspect["aspect_name"] = suggested_name

            aspect["document_coverage"] = aspect_coverage
            aspect["supporting_evidence"] = supporting_evidence
            
            if aspect_coverage < CONFIG["aspects"]["document_coverage_min"]:
                logger.info(f"Aspect '{aspect['aspect_name']}' rejected: {aspect_coverage:.0%} coverage. Missing: {missing_info[:50]}")
                diagnostic["aspects_rejected"] += 1
                diagnostic["rejection_reasons"].append(f"low_coverage_{aspect_coverage:.0%}")
                continue
            
            validated_aspects.append(aspect)
            diagnostic["aspects_validated"] += 1
            logger.info(f"✓ Aspect '{aspect['aspect_name']}' validated ({aspect_coverage:.0%} coverage)")
        
        if len(validated_aspects) >= max_aspects:
            logger.info(f"Reached max aspects ({max_aspects}), stopping extraction.")
            break
            
    if len(validated_aspects) < min_aspects:
        if len(validated_aspects) >= 2 and extraction_attempts >= 3:
            logger.warning(f"Accepting {len(validated_aspects)} aspects after {extraction_attempts} attempts (below minimum {min_aspects})")
            diagnostic["fallback_accepted"] = True
            return validated_aspects, diagnostic
        
        diagnostic["has_issue"] = True
        diagnostic["issue_type"] = f"insufficient_validated_aspects_{len(validated_aspects)}"
        logger.warning(f"Only {len(validated_aspects)} aspects validated (need {min_aspects})")
        return None, diagnostic
    
    diagnostic["success"] = True
    logger.info(f"Successfully extracted {len(validated_aspects)} aspects (min: {min_aspects}, max: {max_aspects})")
    return validated_aspects, diagnostic


# ============================================================
# ORDER ASPECTS BY LOGICAL PROGRESSION
# ============================================================

def order_aspects_by_progression(
    aspects: List[Dict[str, Any]],
    conversation_strategy: str,
    client: AzureOpenAIClient
) -> List[Dict[str, Any]]:
    """Order aspects to follow natural conversation progression."""
    if len(aspects) <= 2:
        return aspects
    
    aspect_summaries = "\n".join([
        f"{i+1}. [{a.get('aspect_type', 'unknown')}] {a['aspect_name']}"
        for i, a in enumerate(aspects)
    ])
    
    prompt = f"""Order these topic aspects for natural conversation progression.

CONVERSATION STRATEGY: {conversation_strategy}

ASPECTS (format: [type] name):
{aspect_summaries}

ORDERING PRINCIPLES:
1. Foundational/definitional aspects first (what something IS)
2. Then mechanisms/processes (how it WORKS)
3. Then historical context or causes (WHY/WHEN)
4. Then specific examples/cases (INSTANCES)
5. End with implications/modern relevance (SO WHAT)

EXAMPLE GOOD ORDER:
- [factual] "Definition of nation vs state" → first (foundational)
- [explanatory] "Distinction between political and cultural entities" → second (mechanism)
- [historical] "Colonial impact on borders" → third (historical cause)
- [factual] "Stateless nations examples" → fourth (examples)
- [explanatory] "Modern nation-state challenges" → last (implications)

EXAMPLE BAD ORDER:
- "Specific case of Kurds" → "What is a nation" (example before definition)
- "Modern implications" → "Basic distinction" (conclusion before foundation)

Return JSON:
{{
  "ordered_indices": [0, 2, 1, 3],
  "reasoning": "Brief explanation of ordering logic"
}}"""
    
    result, error = client.call_gpt_json(
        system_prompt="You order topic aspects for logical conversation flow.",
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"].get("aspect_ordering", 0.2),
        max_tokens=CONFIG["llm"]["max_tokens"].get("aspect_ordering", 300)
    )
    
    if error or not result:
        logger.warning("Aspect ordering failed, using original order")
        return aspects
    
    ordered_indices = result.get("ordered_indices", [])
    
    if not ordered_indices or len(ordered_indices) != len(aspects):
        logger.warning(f"Invalid indices count: {len(ordered_indices)} vs {len(aspects)}")
        return aspects
    
    if set(ordered_indices) != set(range(len(aspects))):
        logger.warning(f"Invalid indices values: {ordered_indices}")
        return aspects
    
    ordered_aspects = [aspects[i] for i in ordered_indices]
    
    logger.info(f"Aspects reordered: {[a['aspect_name'] for a in ordered_aspects]}")
    
    return ordered_aspects


# ============================================================
# GENERATE SUB-QUESTION FOR EACH ASPECT
# ============================================================

SYSTEM_PROMPT_SUBQUESTION = """You generate focused sub-questions that target specific aspects."""

USER_PROMPT_SUBQUESTION = """Generate a focused sub-question for this aspect.

QUERY: {query}

OVERALL REASONING: {overall_reasoning}

ASPECT: {aspect_name}
TYPE: {aspect_type}

KEY FACTS:
{semantic_facts}

PREVIOUS SUB-QUESTIONS:
{previous_subquestions}

Generate a sub-question that:
1. Aligns with the overall reasoning and query intent
2. Specifically targets this aspect
3. Is distinct from previous sub-questions
4. Can be answered using the facts listed

Return JSON:
{{
  "sub_question": "The focused sub-question?",
  "confidence": 0.0-1.0,
  "reasoning": "Why this targets this aspect AND aligns with overall reasoning"
}}"""

def generate_subquestion_for_aspect(
    aspect: Dict[str, Any],
    query: str,
    overall_reasoning: str,
    semantic_facts: List[str],
    previous_subquestions: List[str]
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Generate a sub-question for a specific aspect using semantic facts."""
    client = get_azure_client()
    
    metadata = {"aspect_name": aspect.get("aspect_name")}
    
    facts_text = "\n".join([f"- {fact}" for fact in semantic_facts])
    
    prompt = USER_PROMPT_SUBQUESTION.format(
        query=query,
        overall_reasoning=overall_reasoning,
        aspect_name=aspect.get("aspect_name", ""),
        aspect_type=aspect.get("aspect_type", ""),
        semantic_facts=facts_text,
        previous_subquestions="\n".join([f"- {sq}" for sq in previous_subquestions]) if previous_subquestions else "None"
    )
    
    result, error = client.call_gpt_json(
        system_prompt=SYSTEM_PROMPT_SUBQUESTION,
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"]["subquestion"],
        max_tokens=CONFIG["llm"]["max_tokens"]["subquestion"]
    )
    
    if error:
        metadata["error"] = error
        return None, metadata
    
    if not validate_json_response(result, ["sub_question"], {"sub_question": str}, {"sub_question": 10}):
        metadata["error"] = "validation_failed"
        return None, metadata
    
    metadata["confidence"] = result.get("confidence", 0.0)
    metadata["success"] = True
    
    return result["sub_question"], metadata


# ============================================================
# DOCUMENT ANSWERABILITY CHECK
# ============================================================

def can_documents_answer_question(
    sub_question: str,
    documents: Dict[str, str],
    semantic_facts: List[str],
    client: AzureOpenAIClient
) -> Tuple[bool, float, str]:
    """Check if documents contain enough information to answer the sub-question."""
    all_docs_text = "\n---\n".join([
        f"[{doc_id}]: {content}"
        for doc_id, content in documents.items()
    ])
    
    facts_text = "\n".join([f"- {fact}" for fact in semantic_facts])
    
    min_answerability = CONFIG["turns"]["min_answerability"]
    
    prompt = f"""Can these documents provide a COMPLETE, USEFUL answer to this question?

QUESTION: {sub_question}

KEY FACTS NEEDED:
{facts_text}

DOCUMENTS:
{all_docs_text}

EVALUATION CRITERIA:
- Can documents provide a SUBSTANTIVE answer (not just tangentially related)?
- Would the answer be USEFUL and INFORMATIVE to someone asking this question?
- Are the SPECIFIC details needed to answer present (not just general context)?

Score 0.0-1.0:
- 0.8-1.0: Documents fully answer the question with specific details
- 0.6-0.8: Documents answer most of the question, minor gaps acceptable
- 0.4-0.6: Documents only partially address the question
- 0.0-0.4: Documents lack the specific information needed

Return JSON:
{{
    "can_answer": boolean,
    "coverage_score": 0.0-1.0,
    "reason": "what documents can/cannot address",
    "specific_gaps": ["what's missing"]
}}"""
    
    result, error = client.call_gpt_json(
        system_prompt="You assess if documents can provide useful answers. Be accurate, not generous.",
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"]["answerability"],
        max_tokens=CONFIG["llm"]["max_tokens"]["answerability"]
    )
    
    if error or not result:
        return False, 0.0, "Validation check failed"
    
    can_answer = result.get("can_answer", False)
    coverage = result.get("coverage_score", 0.0)
    reason = result.get("reason", "")
    
    return can_answer, coverage, reason


# ============================================================
# GENERATE AND VALIDATE SUB-QUESTION WITH RETRY
# ============================================================

def generate_and_validate_subquestion(
    aspect: Dict[str, Any],
    query: str,
    overall_reasoning: str,
    documents: Dict[str, str],
    validated_subquestions: List[str],
    client: AzureOpenAIClient,
    max_retries: int
) -> Optional[Tuple[str, List[str]]]:
    """
    Generate sub-question with retry if unanswerable.
    
    Returns: (sub_question, semantic_facts) or None if all retries fail
    """
    semantic_facts = aspect["semantic_facts"]
    
    for attempt in range(max_retries):
        sub_q, sq_meta = generate_subquestion_for_aspect(
            aspect=aspect,
            query=query,
            overall_reasoning=overall_reasoning,
            semantic_facts=semantic_facts,
            previous_subquestions=validated_subquestions
        )
        
        if not sub_q:
            logger.warning(f"Failed to generate sub-question for aspect '{aspect['aspect_name']}'")
            continue
        
        can_answer, coverage, reason = can_documents_answer_question(
            sub_q, documents, semantic_facts, client
        )
        
        if coverage >= CONFIG["turns"]["min_answerability"]:
            logger.info(f"✓ Sub-question validated on attempt {attempt + 1}: {sub_q[:60]}... ({coverage:.0%})")
            return sub_q, semantic_facts
        else:
            logger.info(f"✗ Attempt {attempt + 1}/{max_retries} failed: {reason[:60]}... ({coverage:.0%})")
    
    logger.warning(f"✗ All {max_retries} attempts failed for aspect '{aspect['aspect_name']}'")
    return None


# ============================================================
# SUBQUESTION REASONING GENERATION
# ============================================================

def generate_subquestion_reasoning(
    sub_question: str,
    original_query: str,
    overall_reasoning: str,
    semantic_facts: List[str]
) -> Tuple[str, Dict[str, Any]]:
    """Generate retrieval-focused reasoning for document selection."""
    client = get_azure_client()
    
    facts_text = "\n".join([f"- {fact}" for fact in semantic_facts])
    
    prompt = f"""Generate RETRIEVAL GUIDANCE for finding relevant documents.

ORIGINAL QUERY: {original_query}

OVERALL UNDERSTANDING: {overall_reasoning}

SUB-QUESTION: {sub_question}

FACTS NEEDED: {facts_text}

TASK: Specify what makes a document RELEVANT vs IRRELEVANT for this sub-question.

Provide:
1. **Target Information** (1 sentence): What specific information should documents contain?
2. **Relevance Signals** (2-3 items): What keywords/concepts indicate relevance?
3. **Irrelevance Signals** (2-3 items): What indicates a document is NOT relevant?

Keep CONCISE (under 60 words total).

Return JSON:
{{
  "target_information": "What documents should contain",
  "relevance_signals": ["signal1", "signal2"],
  "irrelevance_signals": ["signal1", "signal2"],
  "retrieval_guidance": "Complete guidance in 1-2 sentences"
}}"""
    
    result, error = client.call_gpt_json(
        system_prompt="Generate retrieval guidance for document selection in conversational IR.",
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"]["subquestion_reasoning"],
        max_tokens=CONFIG["llm"]["max_tokens"]["subquestion_reasoning"]
    )
    
    if error or not result:
        fallback = f"Find documents explaining: {', '.join(semantic_facts)}"
        return fallback, {"fallback": True, "error": error}
    
    reasoning = result.get("retrieval_guidance", "")
    if not reasoning:
        reasoning = f"Documents about: {', '.join(semantic_facts)}"
    
    metadata = {
        "extraction_method": "llm",
        "target_information": result.get("target_information", ""),
        "relevance_signals": result.get("relevance_signals", []),
        "irrelevance_signals": result.get("irrelevance_signals", []),
        "success": True
    }
    
    return reasoning, metadata


# ============================================================
# DOCUMENT SELECTION FALLBACK
# ============================================================

def apply_document_selection_with_fallback(
    ranked_scores: List[Dict[str, Any]],
    valid_docs: List[Tuple[int, str, str]],
    top_k: int,
    threshold: float
) -> Tuple[List[str], Dict[str, Any]]:
    """Select documents with dynamic threshold and no fixed top_k."""
    diagnostic = {
        "has_issue": False,
        "issue_type": None,
        "fallback_applied": False,
        "best_score": None,
        "threshold_used": threshold,
        "dynamic_threshold": None
    }
    
    if not ranked_scores:
        diagnostic["has_issue"] = True
        diagnostic["issue_type"] = "no_valid_scores"
        return [], diagnostic
    
    best_score = ranked_scores[0]["final_score"]
    diagnostic["best_score"] = best_score
    
    dynamic_threshold = max(threshold, best_score * 0.6)
    diagnostic["dynamic_threshold"] = dynamic_threshold
    diagnostic["threshold_used"] = dynamic_threshold
    
    selected_doc_ids = []
    for s in ranked_scores:
        if s["final_score"] >= dynamic_threshold and s["doc_index"] < len(valid_docs):
            doc_id = valid_docs[s["doc_index"]][1]
            selected_doc_ids.append(doc_id)
    
    if not selected_doc_ids and ranked_scores:
        diagnostic["fallback_applied"] = True
        diagnostic["issue_type"] = "no_docs_above_threshold"
        best_idx = ranked_scores[0]["doc_index"]
        if best_idx < len(valid_docs):
            selected_doc_ids.append(valid_docs[best_idx][1])
    
    return selected_doc_ids, diagnostic


# ============================================================
# DOCUMENT SCORING WITH COMPLETENESS
# ============================================================

def score_documents_with_llm(
    sub_question: str,
    retrieval_reasoning: str,
    semantic_facts: List[str],
    candidate_docs: List[Tuple[str, str]],
    top_k: int
) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
    """Score documents (no reuse penalty)."""
    client = get_azure_client()
    
    threshold = CONFIG["documents"]["scoring_threshold"]
    
    diagnostic = {
        "has_issue": False,
        "issue_type": None,
        "fallback_applied": False,
        "best_score": None,
        "threshold_used": threshold
    }
    
    all_scores = []
    
    if not candidate_docs:
        diagnostic["has_issue"] = True
        diagnostic["issue_type"] = "no_candidate_documents"
        return [], diagnostic, all_scores
    
    valid_docs = []
    for i, (doc_id, content) in enumerate(candidate_docs):
        if not content or content.strip() == "":
            continue
        valid_docs.append((i, doc_id, content))
    
    if not valid_docs:
        diagnostic["has_issue"] = True
        diagnostic["issue_type"] = "all_documents_empty"
        return [], diagnostic, all_scores
    
    docs_text = "\n---\n".join([
        f"DOC {i}: {content}"
        for i, _, content in valid_docs
    ])
    
    facts_text = "\n".join([f"- {fact}" for fact in semantic_facts])
    
    prompt = f"""Score documents using retrieval reasoning guidance.

SUB-QUESTION: {sub_question}

RETRIEVAL REASONING (what we're looking for):
{retrieval_reasoning}

KEY FACTS: {facts_text}

DOCUMENTS:
{docs_text}

SCORING (0-10 each):
- answer_support: Does document match what REASONING asks for AND help answer THIS question?
  * Primary criterion: alignment with retrieval reasoning intent
  * Secondary: direct relevance to sub-question
- completeness: Coverage of key facts
- clarity: Information quality
- misleading: Contradicts facts or reasoning

FINAL SCORE = (support × 0.5) + (completeness × 0.3) + (clarity × 0.15) - (misleading × 0.05)

NOTE: Emphasize answer_support (50% weight) - documents must match retrieval reasoning.

Return JSON:
{{
  "document_scores": [
    {{
      "doc_index": 0,
      "answer_support": 0-10,
      "completeness": 0-10,
      "clarity": 0-10,
      "misleading": 0-10,
      "final_score": calculated,
      "reason": "Why relevant + how it matches reasoning",
      "facts_covered": "Which facts"
    }}
  ]
}}"""
    
    result, error = client.call_gpt_json(
        system_prompt="You assess document relevance for specific questions.",
        user_prompt=prompt,
        temperature=CONFIG["llm"]["temperature"]["document_scoring"],
        max_tokens=CONFIG["llm"]["max_tokens"]["document_scoring"]
    )
    
    if error or not result:
        diagnostic["has_issue"] = True
        diagnostic["issue_type"] = error or "no_result"
        return [], diagnostic, all_scores
    
    if not validate_json_response(result, ["document_scores"], {"document_scores": list}):
        diagnostic["has_issue"] = True
        diagnostic["issue_type"] = "validation_failed"
        return [], diagnostic, all_scores
    
    ranked_scores = []
    for score_obj in result.get("document_scores", []):
        if "doc_index" in score_obj and "final_score" in score_obj:
            doc_idx = score_obj["doc_index"]
            
            if doc_idx < len(valid_docs):
                score_obj["doc_id"] = valid_docs[doc_idx][1]
                ranked_scores.append(score_obj)
                
                all_scores.append({
                    "doc_id": score_obj["doc_id"],
                    "doc_index": doc_idx,
                    "answer_support": score_obj.get("answer_support", 0),
                    "completeness": score_obj.get("completeness", 0),
                    "clarity": score_obj.get("clarity", 0),
                    "misleading": score_obj.get("misleading", 0),
                    "final_score": score_obj["final_score"],
                    "reason": score_obj.get("reason", ""),
                    "facts_covered": score_obj.get("facts_covered", "")
                })
    
    ranked_scores.sort(key=lambda x: x["final_score"], reverse=True)
    
    selected_doc_ids, fallback_diagnostic = apply_document_selection_with_fallback(
        ranked_scores=ranked_scores,
        valid_docs=valid_docs,
        top_k=top_k,
        threshold=threshold
    )
    
    diagnostic.update(fallback_diagnostic)
    
    return selected_doc_ids, diagnostic, all_scores


# ============================================================
# CONVERSATIONAL QUERY GENERATION PROMPTS
# ============================================================

SYSTEM_PROMPT_CONV_QUERY_TURN1 = """You transform technical questions into natural, conversational opening questions.

The user is curious about a topic. Generate a question that sounds like a real person asking, not a textbook or quiz."""

USER_PROMPT_CONV_QUERY_TURN1 = """Transform this into a natural opening question.

ORIGINAL TOPIC: {original_query}

TECHNICAL QUESTION: {sub_question}

REQUIREMENTS:
1. Keep the SAME intent and content as the technical question
2. Make it sound conversational and curious
3. START with a brief topic context (3-5 words) like "In [topic]," or "Regarding [topic],"
4. Use simple, direct language
5. Length: 10-20 words maximum

STYLE GUIDELINES:
- MUST begin with topic anchor like: "In [topic], ...", "Regarding [topic], ...", "For [topic], ..."
- Then ask the question naturally
- NO implied prior conversation: avoid "you mentioned", "as discussed"

EXAMPLES:
Topic: Insect behavior | Tech Q: "Explain phototaxis mechanisms"
✓ GOOD: "In insect behavior, why do bugs fly toward light sources?"

Topic: Human breathing | Tech Q: "Describe nasal cycle regulation"
✓ GOOD: "Regarding human breathing, why do we breathe through one nostril at a time?"

Topic: Smell perception | Tech Q: "Explain olfactory receptor diversity"
✓ GOOD: "In smell perception, are there basic smells that combine like RGB colors?"

Return JSON:
{{
  "conversational_query": "The natural opening question WITH topic intro",
  "kept_technical_content": true/false,
  "natural_language_used": true/false
}}"""

SYSTEM_PROMPT_CONV_QUERY = """You transform technical questions into natural conversational follow-ups.

You simulate a curious human in an ongoing conversation. Generate questions with natural variety in phrasing, structure, and tone—like real people talking."""

USER_PROMPT_CONV_QUERY = """Transform this into a natural follow-up question.

TOPIC: {original_query}

RECENT CONVERSATION:
{history}

TECHNICAL QUESTION TO ASK:
{sub_question}

PREVIOUS OPENERS USED: {previous_starters}

--- CORE REQUIREMENTS ---

1. Keep the SAME content/intent as the technical question
2. Sound like a real curious person, not a student or interviewer
3. VARY your opener—never repeat a starting word/phrase from "previous openers"
4. VARY your structure—don't use the same question pattern as recent turns

--- NATURAL CONVERSATION PATTERNS ---

Real humans vary how they ask questions. Mix these styles across turns:

TRANSITION TYPES (illustrative, not exhaustive—find what fits naturally):
A. Direct: "What makes...", "How does...", "Why do...", "When does..."
B. Curious: "I wonder if...", "What about...", "How come..."
C. Confirming: "Does that mean...", "Is that why...", "Would that..."
D. Probing: "But what if...", "Even when...", "What happens if..."
E. Connecting: "And does that...", "Then how...", "Which would mean..."

OPTIONAL HUMAN TOUCHES (use occasionally, not every turn) like:
- Brief reactions: "Interesting—", "Huh,", "Oh,", "Right,"
- Thinking aloud: "Wait,", "Hmm,", "Actually,"
- Casual fragments: "And the timing?", "But at night?"

--- LANGUAGE RULES ---

DO:
- Use simple, everyday words
- Keep under 20 words
- Use pronouns (it, that, this, they) to connect to previous content
- Use contractions naturally (don't, isn't, wouldn't)
- Vary sentence structure across turns

DON'T:
- Repeat any opener from previous turns (not just "So"—any word)
- Use academic language: "regarding", "pertaining to", "aforementioned"
- Use filler transitions: "Building on that", "To continue", "Following up"
- Ask multiple questions with identical structure

--- EXAMPLES (illustrative—create your own natural variations) ---

These show the VARIETY expected, not phrases to copy:

History about insects attracted to light → possible follow-ups:
- "Does the color of light matter?"
- "What about during daytime though?"
- "Huh, why don't they just fly straight into it then?"
- "And all insects do this?"

History about nasal breathing cycles → possible follow-ups:
- "Wait, so one side is always partially blocked?"
- "How long before it switches?"
- "What controls that?"
- "Does being sick change it?"

Note: The above are examples of variety—generate whatever fits naturally while avoiding repetition.

--- ANTI-PATTERNS ---

✗ Repetitive openers (any word, not just "So"):
  Turn 2: "How does X work?"
  Turn 3: "How does Y happen?"
  Turn 4: "How does Z affect this?"

✗ Repetitive structure:
  Turn 2: "Does that mean [X]?"
  Turn 3: "Does that mean [Y]?"
  
✗ Overly formal:
  "Could you elaborate on the mechanisms?"
  "What are the implications of this?"

--- OUTPUT ---

Return JSON:
{{
  "conversational_query": "The natural follow-up question",
  "transition_type": "A/B/C/D/E",
  "uses_natural_language": true/false,
  "references_previous_content": true/false
}}"""
# ============================================================
# ANSWER GENERATION PROMPTS
# ============================================================

SYSTEM_PROMPT_ANSWER = """You answer questions naturally and accurately using provided information.

You are having a conversation with someone curious about a topic. Answer their question using the information available, speaking naturally as if you're explaining to a friend."""

USER_PROMPT_ANSWER = """Answer this question naturally using the information provided.

CONVERSATION SO FAR:
{history}

CURRENT QUESTION:
{query}

AVAILABLE INFORMATION:
{documents}

HOW TO ANSWER:
1. Use ONLY information from the provided text above
2. Speak naturally - explain as if talking to a curious friend
3. Build on what was discussed earlier in the conversation
4. Focus on WHAT YOU CAN EXPLAIN, not what you can't
5. If the full answer isn't available, explain what IS known about the topic
6. Keep answer conversational: 2-4 sentences, clear and direct

IMPORTANT RULES:
✓ State facts directly and naturally
✓ Connect to previous conversation smoothly
✓ Explain using everyday language
✓ If information is partial, focus on what's available and relevant

✗ Do NOT mention "documents", "sources", "the text", or "the information"
✗ Do NOT end with "However, [gaps in information]"
✗ Do NOT add invented examples, names, or details
✗ Do NOT say "I don't know" - instead focus on what IS known

EXAMPLES:

Bad: "The documents indicate that insects exhibit phototaxis. However, the documents do not mention LED wavelengths."
Good: "Insects are naturally drawn to light through a response called phototaxis. Their light-detecting cells respond to different wavelengths, which explains why various light sources attract them."

Bad: "Based on the sources, the nasal cycle is controlled by the autonomic nervous system. The documents do not specify the exact neural pathways."
Good: "The autonomic nervous system controls the switching - specifically, the hypothalamus alternately activates each side, causing one nostril to become more open while the other restricts airflow."

Return JSON:
{{
  "answer": "Your natural, conversational answer",
  "uses_natural_language": true/false,
  "avoids_meta_references": true/false,
  "focuses_on_available_info": true/false
}}"""
# ============================================================
# HISTORY-BASED CONVERSATIONAL TURN GENERATION
# ============================================================

def generate_conversational_turn(
    sub_question: str,
    supporting_doc_ids: List[str],
    documents: Dict[str, str],
    full_history: str,
    semantic_facts: List[str],
    retrieval_reasoning: str,
    turn_number: int,
    original_query: str
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Generate conversational turn with semantic facts guidance."""
    client = get_azure_client()
    
    diagnostic = {
        "has_issue": False,
        "issue_type": None
    }
    
    valid_doc_ids = [doc_id for doc_id in supporting_doc_ids 
                     if doc_id in documents and documents[doc_id].strip()]
    
    if not valid_doc_ids:
        diagnostic["has_issue"] = True
        diagnostic["issue_type"] = "all_documents_empty"
        return None, diagnostic
    
    docs_text = "\n\n".join([
        f"Document {i+1}:\n{documents[doc_id]}"
        for i, doc_id in enumerate(valid_doc_ids)
    ])
    
    is_first_turn = (
        full_history == "No previous conversation." or
        full_history.strip() == "" or
        full_history is None
    )
    
    if is_first_turn:
        logger.info(f"  Using Turn 1 prompts (first turn, turn_number={turn_number})")
        
        query_prompt = USER_PROMPT_CONV_QUERY_TURN1.format(
            sub_question=sub_question,
            original_query=original_query
        )
        
        conv_result, error = client.call_gpt_json(
            system_prompt=SYSTEM_PROMPT_CONV_QUERY_TURN1,
            user_prompt=query_prompt,
            temperature=CONFIG["llm"]["temperature"]["query_turn1"],
            max_tokens=CONFIG["llm"]["max_tokens"]["query"]
        )
    else:
        logger.info(f"  Using Turn 2+ prompts (has history, turn_number={turn_number})")
        
        previous_starters = extract_previous_starters(full_history)
        query_prompt = USER_PROMPT_CONV_QUERY.format(
            history=full_history,
            sub_question=sub_question,
            original_query=original_query,
            previous_starters=previous_starters
        )
        
        conv_result, error = client.call_gpt_json(
            system_prompt=SYSTEM_PROMPT_CONV_QUERY,
            user_prompt=query_prompt,
            temperature=CONFIG["llm"]["temperature"]["query_followup"],
            max_tokens=CONFIG["llm"]["max_tokens"]["query"]
        )
    
    if error or not validate_json_response(conv_result, ["conversational_query"], 
                                           {"conversational_query": str}, {"conversational_query": 10}):
        diagnostic["has_issue"] = True
        diagnostic["issue_type"] = error or "query_validation_failed"
        return None, diagnostic
    
    conversational_query = conv_result["conversational_query"]
    
    answer_prompt = USER_PROMPT_ANSWER.format(
        history=full_history,
        query=conversational_query,
        documents=docs_text
    )
    
    answer_result, error = client.call_gpt_json(
        system_prompt=SYSTEM_PROMPT_ANSWER,
        user_prompt=answer_prompt,
        temperature=CONFIG["llm"]["temperature"]["answer"],
        max_tokens=CONFIG["llm"]["max_tokens"]["answer"]
    )
    
    if error or not validate_json_response(answer_result, ["answer"], 
                                           {"answer": str}, {"answer": 20}):
        diagnostic["has_issue"] = True
        diagnostic["issue_type"] = error or "answer_validation_failed"
        return None, diagnostic
    
    answer = answer_result["answer"]
    
    turn_result = {
        "turn_id": turn_number,
        "conversational_query": conversational_query,
        "answer": answer,
        "supporting_doc_ids": valid_doc_ids,
        "supporting_documents": [
            {"doc_id": doc_id, "content": documents[doc_id]}
            for doc_id in valid_doc_ids
        ],
        "answer_grounded_with_semantic_guidance": True,
        "facts_covered": answer_result.get("facts_covered", ""),
        "limitations": answer_result.get("limitations", ""),
        "is_first_turn": is_first_turn,
        "semantic_facts": semantic_facts
    }
    
    return turn_result, diagnostic


# ============================================================
# VALIDATION
# ============================================================

def validate_turn(turn: Dict[str, Any], documents: Dict[str, str]) -> bool:
    """Validate single turn structure."""
    required = ["turn_id", "conversational_query", "answer", "supporting_doc_ids", "supporting_documents"]
    
    for key in required:
        if key not in turn:
            return False
    
    if not turn["conversational_query"].strip():
        return False
    
    if not turn["answer"].strip():
        return False
    
    if not turn["supporting_doc_ids"]:
        return False
    
    for doc in turn["supporting_documents"]:
        if not doc.get("content", "").strip():
            return False
    
    return True


def validate_example(example: Dict[str, Any]) -> bool:
    """Validate complete example structure."""
    required = ["id", "original_query", "turns", "num_turns", "metadata"]
    
    for key in required:
        if key not in example:
            return False
    
    if example["num_turns"] != len(example["turns"]):
        return False
    
    if example["num_turns"] == 0:
        return False
    
    return True


# ============================================================
# HISTORY SUMMARIZATION FOR LONG CONTEXTS
# ============================================================

def summarize_conversation_history(
    conversation_history: List[Dict[str, str]],
    threshold: int = 5
) -> str:
    """Summarize conversation history when it exceeds threshold turns."""
    if len(conversation_history) <= threshold:
        return "\n".join([
            f"Q: {h['query']}\nA: {h['answer']}"
            for h in conversation_history
        ]) or "No previous conversation."
    
    early_turns = conversation_history[:-2]
    recent_turns = conversation_history[-2:]
    
    early_topics = []
    for i, turn in enumerate(early_turns, 1):
        topic = turn['query']
        early_topics.append(f"Turn {i}: {topic}")
    
    summary_text = "Earlier discussion covered: " + "; ".join(early_topics)
    
    recent_text = "\n".join([
        f"Q: {h['query']}\nA: {h['answer']}"
        for h in recent_turns
    ])
    
    return f"{summary_text}\n\nRecent conversation:\n{recent_text}"

def extract_previous_starters(history_text: str) -> str:
    """Extract first 1-2 words of each query from history to avoid repetition."""
    import re
    starters = []
    for match in re.finditer(r'Q:\s*(\S+(?:\s+\S+)?)', history_text):
        starter = match.group(1).strip().rstrip('?.,!')
        if starter and starter not in starters:
            starters.append(starter)
    return ", ".join(starters) if starters else "None yet"

# ============================================================
# PHRASE-LEVEL REPETITION DETECTION
# ============================================================

def extract_meaningful_phrases(text: str, phrase_length: int = 4) -> set:
    """
    Extract meaningful N-word phrases from text.
    
    Args:
        text: Input text
        phrase_length: Number of words per phrase (default 4)
    
    Returns:
        Set of meaningful phrases
    """
    # Common stop words to filter out meaningless phrases
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'it', 'its', 'they', 'them', 'their'
    }
    
    # Normalize and tokenize
    words = text.lower().split()
    phrases = set()
    
    # Extract N-word sequences
    for i in range(len(words) - phrase_length + 1):
        phrase_words = words[i:i + phrase_length]
        
        # Skip if phrase is entirely stop words
        if all(w in stop_words for w in phrase_words):
            continue
        
        # Skip if phrase has no content words
        content_words = [w for w in phrase_words if w not in stop_words]
        if len(content_words) < 2:  # Need at least 2 content words
            continue
        
        phrase = ' '.join(phrase_words)
        phrases.add(phrase)
    
    return phrases


def check_answer_phrase_repetition(
    new_answer: str,
    previous_answers: List[str],
    max_repeated_phrases: int = 1,
    source_documents: str = ""
) -> Tuple[bool, List[str], float]:
    """
    Check if new answer repeats exact phrases from previous answers.
    
    DYNAMIC DOMAIN-TERM DETECTION:
    - If a repeated phrase appears in source documents, it's domain terminology (ALLOWED)
    - Only flags phrases that are repeated BUT don't appear in documents (lazy writing)
    
    Examples:
    - "are drawn to light" repeated → OK if appears in documents (domain term)
    - "this is because the" repeated → FLAGGED (doesn't appear in documents)
    
    Args:
        new_answer: New answer text
        previous_answers: List of previous answer texts
        max_repeated_phrases: Maximum allowed non-domain repeated phrases
        source_documents: Concatenated text of source documents (for domain term detection)
    
    Returns:
        Tuple of (is_acceptable, non_domain_repeated_phrases, repetition_ratio)
    """
    if not previous_answers:
        return True, [], 0.0
    
    # Extract phrases from new answer
    new_phrases = extract_meaningful_phrases(new_answer, phrase_length=4)
    
    if not new_phrases:
        return True, [], 0.0
    
    # DYNAMIC: Extract phrases from source documents
    doc_phrases = set()
    if source_documents:
        doc_phrases = extract_meaningful_phrases(source_documents, phrase_length=4)
    
    # Find ALL repeated phrases
    all_repeated_phrases = []
    for prev_answer in previous_answers:
        prev_phrases = extract_meaningful_phrases(prev_answer, phrase_length=4)
        overlap = new_phrases & prev_phrases
        all_repeated_phrases.extend(list(overlap))
    
    # Remove duplicates
    all_repeated_phrases = list(set(all_repeated_phrases))
    
    # DYNAMIC FILTERING: Separate domain terms from actual repetition
    # Domain terms: phrases that appear in source documents (technical terminology)
    # Non-domain: phrases that DON'T appear in documents (lazy writing)
    domain_repeated = [p for p in all_repeated_phrases if p in doc_phrases]
    non_domain_repeated = [p for p in all_repeated_phrases if p not in doc_phrases]
    
    # Calculate repetition ratio (only count non-domain repetitions)
    repetition_ratio = len(non_domain_repeated) / len(new_phrases) if new_phrases else 0.0
    
    # Decision based on non-domain repetitions only
    is_acceptable = len(non_domain_repeated) <= max_repeated_phrases
    
    # Log for debugging
    if domain_repeated:
        logger.debug(f"Domain terms repeated (allowed): {', '.join(domain_repeated[:3])}")
    if non_domain_repeated:
        logger.debug(f"Non-domain phrases repeated (counted): {', '.join(non_domain_repeated[:3])}")
    
    return is_acceptable, non_domain_repeated, repetition_ratio


# ============================================================
# CHECK SEMANTIC FACT OVERLAP
# ============================================================

def check_semantic_fact_overlap_with_used_turns(
    new_semantic_facts: List[str],
    used_turns: List[Dict[str, Any]],
    client: Optional[AzureOpenAIClient] = None
) -> Tuple[bool, float]:
    """Check if new semantic facts have significant overlap with already used turns."""
    if not new_semantic_facts or not used_turns:
        return False, 0.0
    
    threshold = CONFIG["turns"]["diversity"].get("semantic_fact_overlap_threshold", 0.80)
    
    all_used_facts = []
    for turn in used_turns:
        if "semantic_facts" in turn and turn["semantic_facts"]:
            all_used_facts.extend(turn["semantic_facts"])
    
    if not all_used_facts:
        return False, 0.0
    
    similarity = compute_semantic_fact_similarity(new_semantic_facts, all_used_facts, client=client)
    
    has_overlap = similarity > threshold
    
    return has_overlap, similarity


# ============================================================
# TURN DIVERSITY VALIDATION
# ============================================================

def validate_turn_diversity(
    new_turn: Dict[str, Any],
    previous_turns: List[Dict[str, Any]],
    use_llm: bool = True
) -> Tuple[bool, str]:
    """Validate turn diversity with optional LLM check."""
    min_new_words = CONFIG["turns"]["diversity"]["min_new_words"]
    max_similarity = CONFIG["turns"]["diversity"]["max_similarity"]
    
    if not previous_turns:
        return True, "First turn"
    
    new_query = new_turn.get("conversational_query", "").lower()
    new_answer = new_turn.get("answer", "").lower()
    
    new_q_words = set(new_query.split())
    new_a_words = set(new_answer.split())
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
    new_q_words -= stop_words
    new_a_words -= stop_words
    
    max_q_similarity = 0.0
    max_a_similarity = 0.0
    
    for i, prev_turn in enumerate(previous_turns, 1):
        prev_query = prev_turn.get("conversational_query", "").lower()
        prev_answer = prev_turn.get("answer", "").lower()
        
        prev_q_words = set(prev_query.split()) - stop_words
        prev_a_words = set(prev_answer.split()) - stop_words
        
        if new_q_words and prev_q_words:
            q_overlap = len(new_q_words & prev_q_words)
            q_similarity = q_overlap / len(new_q_words)
            max_q_similarity = max(max_q_similarity, q_similarity)
            
            if q_similarity > max_similarity:
                return False, f"Question too similar to Turn {i} ({q_similarity:.1%})"
        
        if new_a_words and prev_a_words:
            a_overlap = len(new_a_words & prev_a_words)
            a_unique = len(new_a_words - prev_a_words)
            a_similarity = a_overlap / len(new_a_words)
            max_a_similarity = max(max_a_similarity, a_similarity)
            
            if a_similarity > max_similarity:
                return False, f"Answer too similar to Turn {i} ({a_similarity:.1%})"
            
            if a_unique < min_new_words:
                return False, f"Only {a_unique} new words vs Turn {i}"
    
    # Existing word-level diversity confirmed
    word_level_diverse = True
    
    # NEW: Check phrase-level repetition with dynamic domain detection
    new_answer = new_turn.get("answer", "")
    prev_answers = [t.get("answer", "") for t in previous_turns]
    
    # DYNAMIC: Concatenate source documents from this turn
    source_docs_text = ""
    if "supporting_documents" in new_turn:
        source_docs_text = "".join([
                doc.get("content", "") 
                for doc in new_turn["supporting_documents"]
            ])
    
    phrase_ok, repeated_phrases, repetition_ratio = check_answer_phrase_repetition(
        new_answer, 
        prev_answers, 
        max_repeated_phrases=CONFIG["turns"]["diversity"]["max_repeated_phrases"],
        source_documents=source_docs_text
    )
    
    if not phrase_ok:
        return False, f"Repeated phrases: {', '.join(repeated_phrases[:3])[:100]}"
    
    if repetition_ratio > CONFIG["turns"]["diversity"]["max_phrase_repetition_ratio"]:
        return False, f"High phrase repetition: {repetition_ratio:.0%} of answer phrases are reused"
    
    # If word-level diversity is borderline, use LLM check
    if use_llm and len(previous_turns) >= 2:
        if max_q_similarity > 0.25 or max_a_similarity > 0.25:
            client = get_azure_client()
            is_diverse, llm_reason = validate_turn_diversity_llm(new_turn, previous_turns, client)
            if not is_diverse:
                return False, f"LLM check failed: {llm_reason}"
    
    # Log diversity metrics for monitoring
    if repeated_phrases:
        logger.debug(f"Turn has {len(repeated_phrases)} repeated phrase(s) but within acceptable limit")
    
    return True, f"Diverse (phrase_rep: {repetition_ratio:.0%})"


# ============================================================
# PROCESS TURNS (MODIFIED WITH SOFT GATE)
# ============================================================

def process_turns(
    sub_questions: List[str],
    semantic_facts_list: List[List[str]],
    query: str,
    overall_reasoning: str,
    gold_doc_ids: List[str],
    documents: Dict[str, str],
    docs_per_turn: int,
    idx: int,
    existing_turns_conv: Optional[List[Dict]] = None,
    existing_turns_full: Optional[List[Dict]] = None,
    existing_history: Optional[List[Dict]] = None
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Process sub-questions into turns. Modified to allow deepening (soft overlap check)."""
    client = get_azure_client()
    
    turns_conversational = existing_turns_conv if existing_turns_conv else []
    turns_full = existing_turns_full if existing_turns_full else []
    conversation_history = existing_history if existing_history else []
    problematic_turns = []
    
    for sub_q_idx, (sub_question, semantic_facts) in enumerate(zip(sub_questions, semantic_facts_list)):
        turn_num = len(turns_conversational) + 1
        
        has_fact_overlap, fact_similarity = check_semantic_fact_overlap_with_used_turns(
            semantic_facts, turns_full, client
        )
        
        if has_fact_overlap:
            if fact_similarity > 0.80:
                logger.info(f"  [{idx}] Turn {turn_num} skipped: {fact_similarity:.0%} semantic fact overlap (too similar)")
                problematic_turns.append({
                    "turn_number": turn_num,
                    "issue": f"semantic_fact_overlap_{fact_similarity:.0%}",
                    "solved": False
                })
                continue
            else:
                logger.info(f"  [{idx}] Turn {turn_num} note: {fact_similarity:.0%} fact overlap (allowing for potential deepening)")
        
        retrieval_reasoning, subq_reasoning_metadata = generate_subquestion_reasoning(
            sub_question, query, overall_reasoning, semantic_facts
        )
        
        logger.info(f"  [{idx}] Turn {turn_num} Sub-Q: {sub_question[:80]}... (pre-validated)")
        
        candidate_docs = [(doc_id, documents[doc_id]) for doc_id in gold_doc_ids if doc_id in documents]
        
        supporting_doc_ids, scoring_diagnostic, all_scores = score_documents_with_llm(
            sub_question=sub_question,
            retrieval_reasoning=retrieval_reasoning,
            semantic_facts=semantic_facts,
            candidate_docs=candidate_docs,
            top_k=docs_per_turn
        )
        
        logger.info(f"  [{idx}] Turn {turn_num} Selected {len(supporting_doc_ids)} docs: {supporting_doc_ids}")

        if all_scores:
            logger.info(f"  [{idx}] Turn {turn_num} Scores:")
            for score in all_scores:
                if score["doc_id"] in supporting_doc_ids:
                    logger.info(f"    - {score['doc_id']}: {score['final_score']:.2f}")

        if scoring_diagnostic["has_issue"]:
            problematic_turns.append({
                "turn_number": turn_num,
                "issue": scoring_diagnostic["issue_type"],
                "solved": scoring_diagnostic["fallback_applied"]
            })
            
            if not supporting_doc_ids:
                logger.warning(f"  [{idx}] Turn {turn_num} - no docs")
                continue
        
        threshold = CONFIG["turns"].get("history_summarization_threshold", 5)
        full_history_text = summarize_conversation_history(conversation_history, threshold)
        
        original_sub_question = sub_question
        
        max_diversity_retries = CONFIG["turns"]["diversity"].get("max_retries", 2)
        turn_generated = False
        diversity_retry_count = 0
        
        diversity_reason = "First attempt"
        for diversity_attempt in range(max_diversity_retries + 1):
            if diversity_attempt > 0:
                current_sub_question = f"{original_sub_question}\n\nAVOID: {diversity_reason}. Generate a query focusing on a DIFFERENT aspect or angle."
            else:
                current_sub_question = sub_question
            
            turn_result, turn_diagnostic = generate_conversational_turn(
                sub_question=current_sub_question,
                supporting_doc_ids=supporting_doc_ids,
                documents=documents,
                full_history=full_history_text,
                semantic_facts=semantic_facts,
                retrieval_reasoning=retrieval_reasoning,
                turn_number=turn_num,
                original_query=query
            )
            
            if turn_diagnostic["has_issue"]:
                problematic_turns.append({
                    "turn_number": turn_num,
                    "issue": turn_diagnostic["issue_type"],
                    "solved": False
                })
                logger.warning(f"  [{idx}] Turn {turn_num} generation failed")
                break
            
            if not turn_result or not validate_turn(turn_result, documents):
                problematic_turns.append({
                    "turn_number": turn_num,
                    "issue": "validation_failed",
                    "solved": False
                })
                logger.warning(f"  [{idx}] Turn {turn_num} validation failed")
                break
            
            is_diverse, diversity_reason = validate_turn_diversity(
                turn_result, turns_full, use_llm=True
            )
            
            if is_diverse:
                turn_generated = True
                break
            else:
                diversity_retry_count += 1
                logger.warning(f"  [{idx}] Turn {turn_num} rejected: {diversity_reason} (attempt {diversity_attempt + 1}/{max_diversity_retries + 1})")
                
                if diversity_attempt < max_diversity_retries:
                    logger.info(f"  [{idx}] Retrying turn generation with negative constraints...")
                else:
                    problematic_turns.append({
                        "turn_number": turn_num,
                        "issue": f"diversity: {diversity_reason} (after {diversity_retry_count} retries)",
                        "solved": False
                    })
                    break
        
        if not turn_generated:
            continue
        
        turns_conversational.append({
            "turn_id": turn_result["turn_id"],
            "query": turn_result["conversational_query"],
            "answer": turn_result["answer"]
        })
        
        # Validate answer quality (focuses on available info, not gaps)
        answer_text = turn_result["answer"]
        
        # Check if answer is dominated by "not available" statements
        gap_indicators = [
            "do not mention",
            "do not specify",
            "do not explain",
            "do not address",
            "not mentioned",
            "not specified",
            "not found",
            "don't contain",
            "lack information"
        ]
        
        answer_lower = answer_text.lower()
        has_gap_focus = any(indicator in answer_lower for indicator in gap_indicators)
        
        if has_gap_focus:
            # Log warning but don't reject (answer might still be partially useful)
            logger.warning(f"  [{idx}] Turn {turn_num} answer mentions information gaps")
            quality_note = "mentions_information_gaps"
        else:
            quality_note = "focuses_on_available_information"
        
        turn_full = {
            "turn_id": turn_result["turn_id"],
            "conversational_query": turn_result["conversational_query"],
            "answer": turn_result["answer"],
            "supporting_doc_ids": turn_result["supporting_doc_ids"],
            "supporting_documents": turn_result["supporting_documents"],
            "sub_question": sub_question,
            "semantic_facts": semantic_facts,
            "facts_covered": turn_result.get("facts_covered", ""),
            "limitations": turn_result.get("limitations", ""),
            "subquestion_reasoning": retrieval_reasoning,
            "subquestion_reasoning_metadata": subq_reasoning_metadata,
            "conversation_history_at_turn": full_history_text,
            "quality_note": quality_note,
            "document_scoring": {
                "candidate_doc_count": len(candidate_docs),
                "scores": all_scores,
                "selection_method": "fallback" if scoring_diagnostic.get("fallback_applied") else "dynamic_threshold",
                "best_score": scoring_diagnostic.get("best_score"),
                "dynamic_threshold": scoring_diagnostic.get("dynamic_threshold")
            },
            "turn_diagnostics": {
                "has_issue": scoring_diagnostic["has_issue"] or turn_diagnostic["has_issue"],
                "issue_type": scoring_diagnostic.get("issue_type") or turn_diagnostic.get("issue_type"),
                "fallback_applied": scoring_diagnostic.get("fallback_applied", False),
                "diversity_validated": True,
                "diversity_reason": diversity_reason,
                "diversity_retry_count": diversity_retry_count,
                "fact_overlap_at_entry": fact_similarity if has_fact_overlap else 0.0
            }
        }
        turns_full.append(turn_full)
        
        conversation_history.append({
            "query": turn_result["conversational_query"],
            "answer": turn_result["answer"]
        })
    
    return turns_conversational, turns_full, conversation_history, problematic_turns


# ============================================================
# PROCESS SINGLE EXAMPLE
# ============================================================

def process_single_example(
    example: Dict[str, Any],
    idx: int,
    documents: Dict[str, str],
    gpt4_reasoning_dict: Dict[str, str],
    human_reasoning_dict: Dict[str, str],
    docs_per_turn: int,
    task_split: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Process single example using BATCH aspect extraction + sub-question validation."""
    client = get_azure_client()
    
    example_id = example["id"]
    query = example["query"]
    gold_doc_ids = example["gold_ids"]
    gold_answer = example.get("gold_answer", "")
    
    human_reasoning = human_reasoning_dict.get(example_id, "")
    gpt4_reasoning_full = gpt4_reasoning_dict.get(example_id, "")
    
    logger.info(f"  [{idx}] Processing {example_id}...")
    logger.debug(f"  [{idx}] Gold answer length: {len(gold_answer)} chars")
    logger.debug(f"  [{idx}] Human reasoning length: {len(human_reasoning)} chars")
    
    if not gpt4_reasoning_full:
        logger.warning(f"  [{idx}] Missing GPT-4 reasoning, skipping")
        return None, None
    
    if not gold_answer or len(gold_answer) < 100:
        logger.warning(f"  [{idx}] Missing or too short gold_answer, skipping")
        return None, None
    
    example_documents = {doc_id: documents[doc_id] for doc_id in gold_doc_ids if doc_id in documents}
    
    if not example_documents:
        logger.warning(f"  [{idx}] No documents for {example_id}")
        return None, None
    
    is_aligned, coverage, unsupported = validate_gold_document_alignment_llm(
        gold_answer, example_documents, client
    )
    
    if not is_aligned:
        fallback_coverage = CONFIG["processing"]["alignment_fallback_coverage"]
        if coverage >= fallback_coverage:
            logger.info(f"  [{idx}] Alignment FALLBACK accepted: {coverage:.0%}")
            is_aligned = True
        else:
            logger.warning(f"  [{idx}] Alignment check FAILED: {coverage:.0%}")
            return None, None
    else:
        logger.info(f"  [{idx}] Alignment OK: {coverage:.0%}")
    
    overall_reasoning = gpt4_reasoning_full
    gpt4_sections = extract_gpt4_reasoning_sections(gpt4_reasoning_full)
    
    reasoning_metadata = {
        "source": "gpt4_reasoning_from_dataset",
        "sections": gpt4_sections,
        "human_reasoning": human_reasoning,
        "conversation_strategy": gpt4_sections.get("reasoning_steps", "")
    }
    
    turns_conversational = []
    turns_full = []
    conversation_history = []
    problematic_turns = []
    decomp_attempt = 0
    
    max_decomp = CONFIG["processing"]["max_decomposition_attempts"]
    
    for decomp_attempt in range(1, max_decomp + 1):
        logger.info(f"  [{idx}] Decomposition attempt {decomp_attempt}/{max_decomp}")
        
        aspects, aspect_diagnostic = identify_aspects_with_document_support_iterative_BATCH(
            query=query,
            reasoning=overall_reasoning,
            gold_answer=gold_answer,
            documents=example_documents,
            client=client,
            min_aspects=CONFIG["aspects"]["min_count"],
            max_aspects=CONFIG["aspects"]["max_count"]["batch"],
            max_extraction_attempts=CONFIG["batch"]["extraction_attempts"]
        )
        
        if not aspects or aspect_diagnostic.get("has_issue"):
            logger.warning(f"  [{idx}] Attempt {decomp_attempt}: Aspect failed - {aspect_diagnostic.get('issue_type')}")
            if decomp_attempt == max_decomp:
                return None, None
            continue
        
        logger.info(f"  [{idx}] Identified {len(aspects)} aspects with document support")
        
        aspects_with_facts = [
            aspect for aspect in aspects 
            if aspect.get("semantic_facts") and len(aspect["semantic_facts"]) > 0
        ]

        if not aspects_with_facts:
            logger.warning(f"  [{idx}] No aspects with verified facts")
            if decomp_attempt == max_decomp:
                return None, None
            continue
        
        logger.info(f"  [{idx}] {len(aspects_with_facts)} aspects with verified facts")
        
        conversation_strategy = reasoning_metadata.get("conversation_strategy", "")
        if len(aspects_with_facts) > 2 and conversation_strategy:
            aspects_with_facts = order_aspects_by_progression(
                aspects_with_facts,
                conversation_strategy,
                client
            )
            logger.info(f"  [{idx}] Aspects ordered for natural conversation flow")
        
        validated_subquestions = []
        validated_semantic_facts = []
        failed_aspects = 0
        
        max_retries = CONFIG["subquestions"]["max_generation_retries"]
        
        for aspect in aspects_with_facts:
            result = generate_and_validate_subquestion(
                aspect=aspect,
                query=query,
                overall_reasoning=overall_reasoning,
                documents=example_documents,
                validated_subquestions=validated_subquestions,
                client=client,
                max_retries=max_retries
            )
            
            if result:
                sub_q, semantic_facts = result
                validated_subquestions.append(sub_q)
                validated_semantic_facts.append(semantic_facts)
            else:
                failed_aspects += 1
        
        logger.info(f"  [{idx}] Validated {len(validated_subquestions)}/{len(aspects_with_facts)} sub-questions ({failed_aspects} failed)")
        
        if len(validated_subquestions) < CONFIG["turns"]["min_count"]:
            logger.warning(f"  [{idx}] Only {len(validated_subquestions)} validated sub-questions (need {CONFIG['turns']['min_count']})")
            if decomp_attempt == max_decomp:
                return None, None
            continue
        
        logger.info(f"  [{idx}] Generated {len(validated_subquestions)} validated sub-questions")
        
        if decomp_attempt == 1:
            turns_conversational, turns_full, conversation_history, new_problematic = process_turns(
                sub_questions=validated_subquestions,
                semantic_facts_list=validated_semantic_facts,
                query=query,
                overall_reasoning=overall_reasoning,
                gold_doc_ids=gold_doc_ids,
                documents=documents,
                docs_per_turn=docs_per_turn,
                idx=idx
            )
            problematic_turns.extend(new_problematic)
        else:
            needed_turns = min(CONFIG["turns"]["min_count"] - len(turns_conversational), len(validated_subquestions))
            sub_questions_to_process = validated_subquestions[:needed_turns + 2]
            semantic_facts_to_process = validated_semantic_facts[:needed_turns + 2]
            
            logger.info(f"  [{idx}] Processing {len(sub_questions_to_process)} additional sub-questions")
            
            turns_conversational, turns_full, conversation_history, new_problematic = process_turns(
                sub_questions=sub_questions_to_process,
                semantic_facts_list=semantic_facts_to_process,
                query=query,
                overall_reasoning=overall_reasoning,
                gold_doc_ids=gold_doc_ids,
                documents=documents,
                docs_per_turn=docs_per_turn,
                idx=idx,
                existing_turns_conv=turns_conversational,
                existing_turns_full=turns_full,
                existing_history=conversation_history
            )
            
            problematic_turns.extend(new_problematic)
        
        if len(turns_conversational) >= CONFIG["turns"]["min_count"]:
            logger.info(f"  [{idx}] Successfully reached {len(turns_conversational)} turns")
            break
    
    if len(turns_conversational) < CONFIG["turns"]["min_count"]:
        logger.warning(f"  [{idx}] Only {len(turns_conversational)} turns after {decomp_attempt} attempts")
        return None, None
    
    timestamp = datetime.now().isoformat()
    
    result_conversational = {
        "id": example_id,
        "task": task_split,
        "original_query": query,
        "num_turns": len(turns_conversational),
        "turns": turns_conversational,
        "metadata": {
            "source": "BRIGHT_unified",
            "num_turns": len(turns_conversational),
            "created_at": timestamp,
            "method": "batch_extraction+fact_verification+subquestion_validation+document_aware+history_based+semantic_fact_overlap_check"
        }
    }
    
    result_full = {
        "id": example_id,
        "task": task_split,
        "original_query": query,
        "gold_answer": gold_answer,
        "human_reasoning": human_reasoning,
        "gpt4_reasoning_full": gpt4_reasoning_full,
        "gpt4_reasoning_sections": gpt4_sections,
        "identified_aspects": aspects_with_facts,
        "num_turns": len(turns_full),
        "turns": turns_full,
        "metadata": {
            "gold_doc_count": len(gold_doc_ids),
            "problematic_turns": problematic_turns,
            "total_aspects_identified": len(aspects_with_facts),
            "decomposition_attempts": decomp_attempt,
            "created_at": timestamp,
            "version": "unified_v1",
            "features": "batch_aspect_extraction+FACT_VERIFICATION_CALIBRATED+subquestion_validation_with_retry+document_aware+history_based+semantic_facts+SEMANTIC_FACT_OVERLAP_CHECK",
            "alignment_coverage": coverage,
            "alignment_check_passed": True,
            "config_used": CONFIG
        }
    }
    
    if validate_example(result_conversational) and validate_example(result_full):
        logger.info(f"  [{idx}] SUCCESS: {len(turns_conversational)} turns from {len(aspects_with_facts)} aspects")
        return result_conversational, result_full
    
    logger.warning(f"  [{idx}] Failed final validation")
    return None, None


# ============================================================
# ENTRY CREATION FUNCTIONS
# ============================================================

def create_benchmark_entry(full: Dict[str, Any]) -> Dict[str, Any]:
    """Create benchmark.jsonl entry."""
    return {
        "id": full["id"],
        "task": full["task"],
        "original_query": full["original_query"],
        "original_answer": full.get("gold_answer", ""),
        "turns": [
            {
                "turn_id": turn["turn_id"],
                "query": turn["conversational_query"],
                "answer": turn["answer"],
                "gold_doc_ids": turn["supporting_doc_ids"],
                "subquestion_reasoning": turn.get("subquestion_reasoning", ""),
                "conversation_history": turn.get("conversation_history_at_turn", "")
            }
            for turn in full["turns"]
        ],
        "metadata": {
            "num_turns": full["num_turns"],
            "gold_doc_count": full["metadata"]["gold_doc_count"],
            "version": full["metadata"]["version"],
            "created_at": full["metadata"].get("created_at", "")
        }
    }


def create_analysis_entry(full: Dict[str, Any]) -> Dict[str, Any]:
    """Create analysis.jsonl entry."""
    return {
        "id": full["id"],
        "task": full["task"],
        "original_query": full["original_query"],
        "original_answer": full.get("gold_answer", ""),
        "human_reasoning": full.get("human_reasoning", ""),
        "identified_aspects": [
            {
                "aspect_name": a["aspect_name"],
                "aspect_type": a.get("aspect_type", ""),
                "semantic_facts": a.get("semantic_facts", [])
            }
            for a in full.get("identified_aspects", [])
        ],
        "turns": [
            {
                "turn_id": t["turn_id"],
                "query": t["conversational_query"],
                "answer": t["answer"],
                "gold_doc_ids": t["supporting_doc_ids"],
                "sub_question": t.get("sub_question", ""),
                "semantic_facts": t.get("semantic_facts", []),
                "facts_covered": t.get("facts_covered", ""),
                "limitations": t.get("limitations", ""),
                "conversation_history": t.get("conversation_history_at_turn", "")
            }
            for t in full["turns"]
        ],
        "metadata": full["metadata"]
    }


def extract_documents_from_entry(full: Dict[str, Any], docs_dict: Dict[str, str]) -> None:
    """Extract unique documents from entry into shared dictionary."""
    for turn in full.get("turns", []):
        for doc in turn.get("supporting_documents", []):
            doc_id = doc["doc_id"]
            if doc_id not in docs_dict:
                docs_dict[doc_id] = doc["content"]


# ============================================================
# MAIN CONVERSION
# ============================================================

def convert_bright_conversational(
    task_split: str,
    output_dir: str,
    max_examples: int,
    docs_per_turn: int,
    batch_size: int = 5,
    max_workers: int = 5
) -> None:
    """Convert BRIGHT with immediate checkpoint saving and resume capability."""
    
    print(f"\n{'='*80}")
    print(f"BRIGHT Conversational Conversion - With Checkpointing")
    print(f"Task: {task_split} | Batch: {batch_size} | Workers: {max_workers}")
    print(f"Features: Immediate Saving + Resume from Checkpoint")
    print(f"{'='*80}\n")
    
    logger.info(f"Loading BRIGHT dataset: {task_split}...")
    try:
        examples_ds, documents, gpt4_reasoning_dict, human_reasoning_dict = load_bright_data(task_split)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Initialize checkpoint writer
    writer = CheckpointWriter(output_dir, task_split)
    
    # Load already processed IDs
    processed_ids = load_processed_ids(writer.benchmark_path)
    
    unprocessed_examples = [
        (idx, example)
        for idx, example in enumerate(examples_ds)
        if example["id"] not in processed_ids and idx < max_examples
    ]
    
    total_to_process = len(unprocessed_examples)
    logger.info(f"To process: {total_to_process} (already completed: {len(processed_ids)})")
    
    if total_to_process == 0:
        logger.info("✓ No new examples to process!")
        return
    
    batches = [
        unprocessed_examples[i:i + batch_size]
        for i in range(0, len(unprocessed_examples), batch_size)
    ]
    
    total_valid = 0
    total_skipped = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"BATCH {batch_num}/{len(batches)}: {len(batch)} examples")
            logger.info(f"{'='*80}")
            
            futures = []
            for idx, example in batch:
                future = executor.submit(
                    process_single_example,
                    example=example,
                    idx=idx,
                    documents=documents,
                    gpt4_reasoning_dict=gpt4_reasoning_dict,
                    human_reasoning_dict=human_reasoning_dict,
                    docs_per_turn=docs_per_turn,
                    task_split=task_split
                )
                futures.append((idx, example["id"], future))
            
            # Process results as they complete
            for idx, example_id, future in futures:
                try:
                    result_conv, result_full = future.result()
                    
                    if result_conv and result_full:
                        # Write immediately
                        if writer.write_record(result_full):
                            total_valid += 1
                            logger.info(f"  ✓ [{idx}] {example_id} - SAVED")
                        else:
                            total_skipped += 1
                            logger.error(f"  ✗ [{idx}] {example_id} - WRITE FAILED")
                    else:
                        total_skipped += 1
                        logger.warning(f"  ✗ [{idx}] {example_id} - PROCESSING FAILED")
                        
                except Exception as e:
                    total_skipped += 1
                    logger.error(f"  ✗ [{idx}] Exception: {e}")
            
            logger.info(f"[OK] Batch {batch_num} complete: {total_valid} valid, {total_skipped} skipped")
            logger.info(f"Unique documents so far: {writer.get_unique_doc_count()}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"[OK] Conversion Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Valid: {total_valid} | Skipped: {total_skipped}")
    logger.info(f"Total completed: {len(processed_ids) + total_valid}")
    logger.info(f"Unique documents: {writer.get_unique_doc_count()}")
    logger.info(f"\nFiles created:")
    logger.info(f"  1. Benchmark:  {writer.benchmark_path}")
    logger.info(f"  2. Analysis:   {writer.analysis_path}")
    logger.info(f"  3. Debug:      {writer.debug_path}")
    logger.info(f"  4. Documents:  {writer.docs_path}")
    logger.info(f"{'='*80}\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    validate_config(CONFIG)
    
    TASKS_SPLIT = ["biology", "psychology", "economics", "earth_science", "sustainable_living", "robotics"]
    OUTPUT_DIR = "outputs_bright_unified"
    MAX_EXAMPLES = 120
    DOCS_PER_TURN = CONFIG["documents"]["per_turn_default"]
    BATCH_SIZE = CONFIG["processing"]["batch_size_default"]
    MAX_WORKERS = CONFIG["processing"]["max_workers_default"]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for task in TASKS_SPLIT:
        logger.info(f"\n{'#'*80}")
        logger.info(f"Starting task: {task}")
        logger.info(f"{'#'*80}")
        
        convert_bright_conversational(
            task_split=task,
            output_dir=OUTPUT_DIR,
            max_examples=MAX_EXAMPLES,
            docs_per_turn=DOCS_PER_TURN,
            batch_size=BATCH_SIZE,
            max_workers=MAX_WORKERS
        )
    
    logger.info(f"\n{'#'*80}")
    logger.info(f"ALL TASKS COMPLETED!")
    logger.info(f"{'#'*80}")
    
    print("\n [OK] Complete! Check outputs_bright_unified/ directory.")