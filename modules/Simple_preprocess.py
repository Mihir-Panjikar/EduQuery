import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

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
    text = re.sub(r'•|\uf071|◉', '- ', text)  # Replaces various bullet styles
    return text


def fix_hyphenated_words(text):
    """Fixes words that are broken across lines with hyphens."""
    return re.sub(r'-\n', '', text)  # Removes hyphen & newline to merge words


def clean_text(text):
    """Removes unwanted spaces, newlines, and special characters."""
    text = re.sub(r"\n+", "\n", text)  # Remove extra newlines
    text = re.sub(r" {2,}", " ", text)  # Remove extra spaces
    text = re.sub(r"—|-", "", text)  # Remove unnecessary dashes
    return text.strip()


def chunk_text(text, max_chunk_size=400):
    """Splits text into meaningful chunks while preserving structure."""
    sentences = sent_tokenize(text)
    chunks = []
    buffer = []
    buffer_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if buffer_len + sentence_len > max_chunk_size:
            chunks.append(" ".join(buffer))
            buffer = []
            buffer_len = 0  # Reset buffer length

        buffer.append(sentence)
        buffer_len += sentence_len

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks


def preprocess_text(text):
    """Applies all preprocessing steps to clean text."""
    text = remove_html_tags(text)
    text = remove_url(text)
    text = normalize_bullet_points(text)
    text = fix_hyphenated_words(text)
    text = clean_text(text)
    return text
