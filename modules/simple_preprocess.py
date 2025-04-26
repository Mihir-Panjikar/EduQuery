import re
import nltk
from nltk.tokenize import sent_tokenize
import logging
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    text = re.sub(r"\n{3,}", "\n\n",
                  text)  # Reducing multiple newlines to max two
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


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Splits text into chunks using Langchain's RecursiveCharacterTextSplitter.

    Args:
        text: The input text.
        chunk_size: The target size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if not text or not isinstance(text, str):
        logger.warning(
            "Invalid input text to chunk_text, returning empty list.")
        return []


    separators = ["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        add_start_index=False,
    )

    try:
        chunks = text_splitter.split_text(text)

        return [chunk for chunk in chunks if chunk and not chunk.isspace()]
    except Exception as e:
        logger.error(f"Error during text splitting with Langchain: {e}")

        return []
