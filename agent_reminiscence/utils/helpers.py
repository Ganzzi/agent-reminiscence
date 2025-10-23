"""
Utility helper functions for Agent Mem.

Provides text processing, chunking, and other utility operations.
"""

import logging
from typing import List, Any
import re

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    preserve_sentences: bool = True,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk (default: 512)
        overlap: Characters to overlap between chunks (default: 50)
        preserve_sentences: Try to break at sentence boundaries (default: True)

    Returns:
        List of text chunks

    Example:
        chunks = chunk_text("Long text here...", chunk_size=100, overlap=20)
        print(f"Created {len(chunks)} chunks")
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # If text is shorter than chunk_size, return as single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []

    if preserve_sentences:
        # Split by sentences (rough heuristic)
        sentences = re.split(r"([.!?]+\s+)", text)

        # Recombine sentences with their punctuation
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                combined_sentences.append(sentences[i] + sentences[i + 1])
            else:
                combined_sentences.append(sentences[i])

        # Build chunks from sentences
        current_chunk = ""
        for sentence in combined_sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                if chunks and overlap > 0:
                    # Take last 'overlap' characters from previous chunk
                    overlap_text = chunks[-1][-overlap:]
                    current_chunk = overlap_text + sentence
                else:
                    current_chunk = sentence

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

    else:
        # Simple character-based chunking
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap if overlap > 0 else end

    logger.debug(
        f"Chunked text ({len(text)} chars) into {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={overlap})"
    )

    return chunks


def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalize a vector to unit length.

    Args:
        vector: Vector to normalize

    Returns:
        Normalized vector

    Example:
        normalized = normalize_vector([1.0, 2.0, 3.0])
    """
    magnitude = sum(x * x for x in vector) ** 0.5

    if magnitude == 0:
        logger.warning("Cannot normalize zero vector")
        return vector

    return [x / magnitude for x in vector]


def parse_external_id(external_id: Any) -> str:
    """
    Parse external_id to string format.

    Handles UUID, int, string, and other types.

    Args:
        external_id: External ID of any type

    Returns:
        String representation of external_id

    Example:
        external_id_str = parse_external_id(uuid.uuid4())
        external_id_str = parse_external_id(12345)
    """
    return str(external_id)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length (default: 100)
        suffix: Suffix to add if truncated (default: "...")

    Returns:
        Truncated text

    Example:
        short = truncate_text("Very long text here...", max_length=20)
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing newlines.

    Args:
        text: Text to clean

    Returns:
        Cleaned text

    Example:
        cleaned = clean_text("Text   with    extra   spaces\\n\\n\\n")
    """
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)

    # Replace multiple newlines with double newline
    text = re.sub(r"\n\n+", "\n\n", text)

    # Strip leading/trailing whitespace
    return text.strip()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0 to 1)

    Example:
        similarity = cosine_similarity(embedding1, embedding2)
        print(f"Similarity: {similarity:.3f}")
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vectors must have same dimension: {len(vec1)} != {len(vec2)}")

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def merge_metadata(
    existing: dict,
    updates: dict,
    overwrite: bool = False,
) -> dict:
    """
    Merge metadata dictionaries.

    Args:
        existing: Existing metadata
        updates: New metadata to merge
        overwrite: If True, overwrite existing values (default: False)

    Returns:
        Merged metadata dictionary

    Example:
        merged = merge_metadata(
            {"key1": "value1"},
            {"key2": "value2"},
        )
    """
    if not existing:
        return updates.copy()

    if not updates:
        return existing.copy()

    result = existing.copy()

    for key, value in updates.items():
        if key not in result or overwrite:
            result[key] = value

    return result


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text (simple word frequency approach).

    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords (default: 10)

    Returns:
        List of keywords

    Example:
        keywords = extract_keywords("Python is great. Python is powerful.")
        print(keywords)  # ['python', 'great', 'powerful', ...]
    """
    # Lowercase and split into words
    words = re.findall(r"\b\w+\b", text.lower())

    # Filter out common stop words (simple list)
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
    }

    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]

    # Count word frequency
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # Return top keywords
    return [word for word, freq in sorted_words[:max_keywords]]


