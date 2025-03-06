import os
import streamlit as st
import faiss
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from modules.text_extraction import extract_text_from_pdf, extract_text_from_docx, extract_text_from_ppt
from modules.Simple_preprocess import chunk_text


# Load Model (Offline)
MODEL_PATH = "models/all-MiniLM-L6-v2"
if not os.path.exists(MODEL_PATH):
    st.error("Embedding model not found! Please download and save it in 'models/'.")
    st.stop()

model = SentenceTransformer(MODEL_PATH)

# Paths
DATA_FOLDER = "CMS_notes"
FAISS_INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "text_chunks.json"
BM25_FILE = "bm25_index.json"


# Process All Files in Folder
def process_knowledge_base():
    chunks = []
    for file in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, file)
        if file.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        elif file.endswith(".pptx"):
            text = extract_text_from_ppt(file_path)
        else:
            continue

        # Split text into smaller chunks (sentences instead of paragraphs)
        sentences = text.split(". ")  # Splitting at sentence level
        chunk = chunk_text(sentences)
        chunks.extend(chunk)

    # Encode & Store in FAISS
    embeddings = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))

    # Save embeddings & text chunks
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunks, f)

    # BM25 Indexing
    tokenized_corpus = [chunk.split(" ") for chunk in chunks]  # Tokenize
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_FILE, "w") as f:
        json.dump({"chunks": chunks}, f)

# Retrieve Answer from FAISS


def get_answer(query):
    query_embedding = model.encode([query], convert_to_numpy=True)
    index = faiss.read_index(FAISS_INDEX_FILE)

    with open(CHUNKS_FILE, "r") as f:
        text_chunks = json.load(f)

    with open(BM25_FILE, "r") as f:
        bm25_data = json.load(f)
        bm25_corpus = [chunk.split(" ") for chunk in bm25_data["chunks"]]
        bm25 = BM25Okapi(bm25_corpus)

    # FAISS Retrieval
    query_embedding = model.encode([query], convert_to_numpy=True)
    _, faiss_idx = index.search(query_embedding, 1)  # Get top 1 matches
    faiss_results = [text_chunks[i] for i in faiss_idx[0]]

    # BM25 Retrieval
    bm25_scores = bm25.get_scores(query.split(" "))
    bm25_top_indexes = np.argsort(bm25_scores)[-1:][::-1]  # Get top 1
    bm25_results = [text_chunks[i] for i in bm25_top_indexes]

    # Merge Results (FAISS + BM25)
    combined_results = list(
        set(faiss_results + bm25_results))  # Remove duplicates
    return " ".join(combined_results)  # Return combined best-matched text


# Streamlit UI
st.title("📚 AI-Powered Q&A System (Offline)")
st.subheader("Ask questions based on the predefined knowledge base!")

# Preprocess Knowledge Base (Only Once)
if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(CHUNKS_FILE) or os.path.exists(BM25_FILE):
    st.info("Processing knowledge base...")
    process_knowledge_base()
    st.success("Knowledge base processed successfully!")


# Question Input
query = st.text_input("Ask a question:")
if query:
    answer = get_answer(query.lower())
    if answer:
        st.code(answer, language="markdown")
