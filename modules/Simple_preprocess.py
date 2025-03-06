import re
import nltk
from nltk.tokenize import sent_tokenize

# Ensure you have the necessary resources for tokenization
nltk.download('punkt')


def remove_html_tags(text):
    pattern = re.compile(r'<[^>]+>', re.MULTILINE)
    return pattern.sub(r'', text)


def lower_case(text):
    return text.lower()


def remove_url(text):
    pattern = re.compile(r'\n?(?:https?|HTTP)://\S+|www\.\S+', re.IGNORECASE)
    return pattern.sub(r'', text)


def clean_text(text):
    text = re.sub(r"\n+", "\n", text)  # Remove multiple newlines
    text = re.sub(r" {2,}", " ", text)  # Remove extra spaces
    text = re.sub(r"—|-", "", text)  # Remove unnecessary dashes
    return text.strip()


def chunk_text(text, max_chunk_size=400):
    sentences = sent_tokenize(" ".join(text))
    chunks = []
    buffer = []
    buffer_len = 0  # Track character length of buffer

    for sentence in sentences:
        sentence_len = len(sentence)

        if buffer_len + sentence_len > max_chunk_size:
            chunks.append(" ".join(buffer))
            buffer = []
            buffer_len = 0  # Reset buffer length

        buffer.append(sentence)
        buffer_len += sentence_len

    # Add remaining text
    if buffer:
        chunks.append(" ".join(buffer))

    return chunks


def preprocess_text(text):
    text = remove_html_tags(text)
    text = remove_url(text)
    text = lower_case(text)
    text = clean_text(text)
    return text
