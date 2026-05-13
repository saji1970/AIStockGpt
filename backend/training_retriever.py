"""
Training Data Retriever for AI Stock GPT
==========================================

Loads JSONL training examples and uses sentence-transformers embeddings
to find the most relevant example for a given user query via cosine similarity.
This enables retrieval-augmented few-shot learning for the LLM provider.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed, training retriever disabled")


class TrainingRetriever:
    """Retrieves the most relevant training example for a user query."""

    def __init__(self, model=None, training_dir: Optional[str] = None):
        """
        Args:
            model: Optional SentenceTransformer instance (shared with nlp_enhanced).
            training_dir: Path to directory containing .jsonl training files.
        """
        self.examples: List[Dict[str, str]] = []
        self.embeddings = None
        self.model = None
        self.ready = False

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("Training retriever disabled: sentence-transformers not available")
            return

        # Resolve training directory
        if training_dir is None:
            base = Path(__file__).resolve().parent.parent
            training_dir = str(base / "training")

        # Load training examples
        self._load_examples(training_dir)
        if not self.examples:
            logger.warning("No training examples found, retriever disabled")
            return

        # Use shared model or load a new one
        if model is not None:
            self.model = model
            logger.info("Training retriever using shared SentenceTransformer model")
        else:
            try:
                self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                logger.info("Training retriever loaded its own SentenceTransformer model")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer for retriever: {e}")
                return

        # Embed all training user queries
        self._embed_examples()

    def _load_examples(self, training_dir: str):
        """Load all .jsonl files from the training directory."""
        training_path = Path(training_dir)
        if not training_path.exists():
            logger.warning(f"Training directory not found: {training_dir}")
            return

        for jsonl_file in sorted(training_path.glob("*.jsonl")):
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            example = json.loads(line)
                            if "user" in example and "assistant" in example:
                                self.examples.append(example)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in {jsonl_file.name} line {line_num}")
            except Exception as e:
                logger.error(f"Error reading {jsonl_file}: {e}")

        logger.info(f"Loaded {len(self.examples)} training examples from {training_dir}")

    def _embed_examples(self):
        """Embed all training example user queries into a numpy matrix."""
        if not self.model or not self.examples:
            return

        user_texts = [ex["user"] for ex in self.examples]
        try:
            self.embeddings = self.model.encode(user_texts, convert_to_numpy=True)
            self.ready = True
            logger.info(f"Embedded {len(user_texts)} training examples (shape: {self.embeddings.shape})")
        except Exception as e:
            logger.error(f"Failed to embed training examples: {e}")

    def find_best_example(
        self, query: str, top_k: int = 1, min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find the most relevant training example(s) for a query.

        Args:
            query: User query string.
            top_k: Number of top examples to return.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of dicts with keys: user, assistant, similarity.
        """
        if not self.ready or self.embeddings is None:
            return []

        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            # Compute cosine similarities
            similarities = cos_sim(query_embedding, self.embeddings)[0].numpy()

            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                sim = float(similarities[idx])
                if sim < min_similarity:
                    continue
                results.append({
                    "user": self.examples[idx]["user"],
                    "assistant": self.examples[idx]["assistant"],
                    "similarity": sim,
                })

            if results:
                logger.info(
                    f"Training retriever: best match for '{query[:50]}...' "
                    f"-> similarity={results[0]['similarity']:.3f}"
                )

            return results

        except Exception as e:
            logger.error(f"Training retriever error: {e}")
            return []

    def set_model(self, model):
        """Set or update the SentenceTransformer model and re-embed if needed."""
        self.model = model
        if self.examples and not self.ready:
            self._embed_examples()
