import re
import nltk
from nltk.tokenize import sent_tokenize
import logging
from typing import List

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)


def remove_html_tags(text):
    """Removes HTML tags from the text."""
    pattern = re.compile(r'<[^>]+>', re.MULTILINE)
    return pattern.sub(r'', text)


def remove_url(text):
    """Removes URLs, including those with leading spaces or line breaks."""
    pattern = re.compile(r'\s*(https?://\S+|www\.\S+)', re.IGNORECASE)
    return pattern.sub(r'', text)


def normalize_bullet_points(text):
    """Replaces different bullet points with a standard '-' format."""
    text = re.sub(r'•|\uf071|◉', '- ', text)
    # text = re.sub(r'^\s*(\d+\.|[a-z]\))\s+', '- ', text, flags=re.MULTILINE)
    return text


def fix_hyphenated_words(text):
    """Fixes words that are broken across lines with hyphens."""
    return re.sub(r'-\n(\s)*', '', text)


def clean_text(text):
    """Removes unwanted spaces, newlines, and less critical characters."""
    text = re.sub(r"\n{3,}", "\n\n", text)  # Reducing multiple newlines to max two
    text = re.sub(r" {2,}", " ", text)  # Removing extra spaces
    text = re.sub(r"—", " - ", text)  # Replacing em-dash
    return text.strip()


def preprocess_text(text):
    """Applies all preprocessing steps to clean text."""
    if not isinstance(text, str):
        logger.warning(
            "Input to preprocess_text is not a string, returning empty string.")
        return ""
    text = remove_html_tags(text)
    text = remove_url(text)
    text = fix_hyphenated_words(text)
    text = normalize_bullet_points(text)
    text = clean_text(text)
    return text


# --- Recursive Character Text Splitting ---

def _split_text_with_separator(text: str, separator: str) -> List[str]:
    """Splits text by separator, keeping the separator with the preceding part."""
    parts = []
    current_pos = 0
    while True:
        idx = text.find(separator, current_pos)
        if idx == -1:
            parts.append(text[current_pos:])
            break
        parts.append(text[current_pos: idx + len(separator)])
        current_pos = idx + len(separator)
    # Filtering out empty strings that might result from consecutive separators
    return [p for p in parts if p]


def _recursive_split(text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """Recursively splits text based on separators."""
    final_chunks = []
    if not text:
        return []

    separator = separators[0]
    remaining_separators = separators[1:]

    splits = _split_text_with_separator(text, separator)

    current_chunk = ""
    for i, part in enumerate(splits):
        # If adding the current part doesn't exceed chunk_size
        if len(current_chunk) + len(part) <= chunk_size:
            current_chunk += part
        else:
            # If the part itself is larger than chunk_size, trying to split it further
            if len(part) > chunk_size:
                if remaining_separators:
                    # Recursing with the next separator
                    sub_chunks = _recursive_split(
                        part, remaining_separators, chunk_size, chunk_overlap)
                    final_chunks.extend(sub_chunks)
                else:
                    # Cannot split further, adding the large part as is (or chunk forcefully)
                    # Forceful chunking (sliding window on the large part)
                    start = 0
                    while start < len(part):
                        end = min(start + chunk_size, len(part))
                        final_chunks.append(part[start:end])
                        start += chunk_size - chunk_overlap
                        if start >= len(part):  # Avoiding infinite loop if overlap >= size
                            break

            if current_chunk:
                final_chunks.append(current_chunk)

            # Start the new chunk with overlap from the previous chunk if possible
            overlap_text = current_chunk[max(
                0, len(current_chunk) - chunk_overlap):] if current_chunk else ""

            # If the current part fits within the size limit *with overlap*
            if len(overlap_text) + len(part) <= chunk_size:
                current_chunk = overlap_text + part
            else:
                # If the part itself is too big even with overlap, starting fresh
                if len(part) <= chunk_size:
                    current_chunk = part  # Start new chunk only with this part
                else:
                    current_chunk = ""  # Resetting as sub-chunks were added directly

    # Add the last remaining chunk
    if current_chunk:
        final_chunks.append(current_chunk)

    return final_chunks


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Splits text into chunks using recursive character splitting.

    Args:
        text: The input text.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if not text:
        return []

    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    initial_splits = [text]
    if len(text) > chunk_size:
        primary_separator = separators[0]
        initial_splits = _split_text_with_separator(text, primary_separator)

    # Processing splits recursively and merging it appropriately
    final_chunks = []
    buffer = ""
    for split in initial_splits:
        if len(split) <= chunk_size:
            # If the split fits with the buffer, adding it
            if len(buffer) + len(split) <= chunk_size:
                buffer += split
            else:
                if buffer:
                    final_chunks.append(buffer)
                # Starting new buffer, considering overlap
                overlap_text = buffer[max(
                    0, len(buffer) - chunk_overlap):] if buffer else ""
                if len(overlap_text) + len(split) <= chunk_size:
                    buffer = overlap_text + split
                else:
                    buffer = split  # Starting fresh if split is too large even with overlap
        else:
            # If the initial split is still too large, apply recursive splitting
            if buffer:
                final_chunks.append(buffer)
                buffer = ""  # Reset buffer

            sub_chunks = _recursive_split(
                split, separators, chunk_size, chunk_overlap)
            final_chunks.extend(sub_chunks)

    if buffer:
        final_chunks.append(buffer)

    # Filter out potentially empty or whitespace-only chunks
    return [chunk for chunk in final_chunks if chunk and not chunk.isspace()]
