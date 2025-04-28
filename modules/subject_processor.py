import os
import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import logging
from typing import List, Dict, Optional, Any
from .text_extraction import extract_text_from_pdf, extract_text_from_docx, extract_text_from_ppt
from .simple_preprocess import chunk_text

logger = logging.getLogger(__name__)


def get_available_subjects(data_folder: str) -> List[str]:
    """
    Dynamically discover subjects by scanning folder names in the data directory.
    """
    subjects = []
    try:
        if not os.path.exists(data_folder):
            logger.warning(f"Data folder '{data_folder}' does not exist")
            os.makedirs(data_folder, exist_ok=True)
            return subjects

        for item in os.listdir(data_folder):
            item_path = os.path.join(data_folder, item)
            if os.path.isdir(item_path):
                has_files = False
                for _, _, files in os.walk(item_path):
                    if files:
                        has_files = True
                        break

                if has_files:
                    subjects.append(item)
                else:
                    logger.info(f"Skipping empty subject folder: {item}")
    except Exception as e:
        logger.error(f"Error discovering subjects: {e}")

    return sorted(subjects)


def extract_document_text(file_path: str) -> str:
    """Extract text from document based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext in ['.pptx', '.ppt']:
        return extract_text_from_ppt(file_path)
    else:
        logger.warning(f"Unsupported file type: {ext} in {file_path}")
        return ""


def process_subject_knowledge_base(data_folder: str, indices_folder: str, subject: str, model) -> bool:
    """
    Process all document files for a specific subject and build knowledge base.
    """
    subject_folder = os.path.join(
        data_folder, subject)
    subject_indices_folder = os.path.join(indices_folder, subject)

    os.makedirs(subject_indices_folder, exist_ok=True)

    faiss_index_file = os.path.join(subject_indices_folder, "faiss_index.idx")
    chunks_file = os.path.join(subject_indices_folder, "chunks.json")
    bm25_file = os.path.join(subject_indices_folder, "bm25.json")

    chunks = []
    sources = []

    try:
        for root, _, files in os.walk(subject_folder):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, subject_folder)

                if file.startswith('.'):
                    continue

                logger.info(f"Processing file: {relative_path}")

                # Extracting text based on file type
                if file.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                elif file.lower().endswith('.docx'):
                    text = extract_text_from_docx(file_path)
                elif file.lower().endswith(('.pptx', '.ppt')):
                    text = extract_text_from_ppt(file_path)
                else:
                    logger.warning(f"Unsupported file format: {file}")
                    continue

                if not text:
                    logger.warning(f"No text extracted from {relative_path}")
                    continue

                text_chunks = chunk_text(text)

                if text_chunks:
                    chunks.extend(text_chunks)
                    sources.extend([relative_path] * len(text_chunks))

        if not chunks:
            logger.warning(f"No content extracted for subject: {subject}")
            return False

        # Adjusting batch size based on GPU memory
        batch_size = 8
        all_embeddings = []
        logger.info(
            f"Encoding {len(chunks)} chunks in batches of {batch_size}...")
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            try:
                # Ensure model is on the correct device (redundant if loaded correctly, but safe)
                # device = next(model.parameters()).device
                # batch_embeddings = model.encode(batch_chunks, device=device)
                batch_embeddings = model.encode(batch_chunks)
                all_embeddings.append(batch_embeddings)
                logger.debug(
                    f"Encoded batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            except Exception as batch_e:
                logger.error(
                    f"Error encoding batch starting at index {i}: {batch_e}")

                raise batch_e

        if not all_embeddings:
            logger.error(
                "No embeddings were generated, possibly due to errors in all batches.")
            return False

        embeddings = np.concatenate(all_embeddings, axis=0)

        embeddings = embeddings.astype('float32')

        # Create FAISS index with proper dimensionality
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)

        logger.info(f"Adding {len(embeddings)} embeddings to FAISS index.")

        index.add(embeddings)
        logger.info("Embeddings added successfully.")

        faiss.write_index(index, faiss_index_file)

        # Save chunks and metadata
        with open(chunks_file, "w") as f:
            json.dump({
                "chunks": chunks,
                "sources": sources
            }, f)

        # BM25 index (just reuse the same chunks data at query time)
        # No need to save BM25 object separately since it's quick to recreate
        # and we already have the chunks data saved
        logger.info(
            f"Successfully created knowledge base for {subject} with {len(chunks)} chunks")
        return True

    except Exception as e:
        logger.error(f"Error building knowledge base for {subject}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def calculate_relevance(chunk: str, query: str, model) -> float:
    """Calculate semantic relevance between chunk and query"""
    try:
        chunk_embedding = model.encode([chunk], convert_to_numpy=True)[0]
        query_embedding = model.encode([query], convert_to_numpy=True)[0]
        similarity = np.dot(chunk_embedding, query_embedding) / (
            np.linalg.norm(chunk_embedding) * np.linalg.norm(query_embedding)
        )
        return float(similarity)
    except Exception:
        return 0.5


def get_answer_for_subject(query: str, subject: str, indices_folder: str, model) -> Optional[List[Dict[str, Any]]]:
    """Get answer for a specific subject, returning structured results."""
    subject_indices_folder = os.path.join(indices_folder, subject)
    faiss_index_file = os.path.join(subject_indices_folder, "faiss_index.idx")
    chunks_file = os.path.join(subject_indices_folder, "chunks.json")
    bm25_file = os.path.join(subject_indices_folder, "bm25.json")

    if not os.path.exists(faiss_index_file) or not os.path.exists(chunks_file):
        logger.error(f"Knowledge base for {subject} not found")
        return None

    try:
        with open(chunks_file, "r") as f:
            data = json.load(f)
            text_chunks = data["chunks"]
            sources = data.get("sources", ["unknown"] * len(text_chunks))

        # FAISS retrieval
        query_embedding = model.encode([query], convert_to_numpy=True)
        index = faiss.read_index(faiss_index_file)

        k_initial = 5
        D, faiss_idx = index.search(query_embedding.astype(
            'float32'), k_initial)

        # BM25 retrieval
        all_idx = faiss_idx[0].tolist()
        if os.path.exists(bm25_file):
            with open(bm25_file, "r") as f:
                bm25_data = json.load(f)
            tokenized_corpus = [chunk.split(" ") for chunk in text_chunks]
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(query.split(" "))

            bm25_top_idx = np.argsort(bm25_scores)[-k_initial:][::-1]

            all_idx = list(set(all_idx + bm25_top_idx.tolist()))

        # Re-ranking
        scored_results_data = []
        for idx in all_idx:
            if 0 <= idx < len(text_chunks):
                chunk = text_chunks[idx]
                source = sources[idx]
                relevance_score = calculate_relevance(
                    chunk, query, model)
                scored_results_data.append({
                    "id": f"chunk_{idx}",
                    "text": chunk,
                    "source": source,
                    "score": relevance_score
                })

        # Sorting by relevance score
        scored_results_data.sort(key=lambda x: x["score"], reverse=True)

        final_results = scored_results_data[:5]

        return final_results if final_results else None

    except Exception as e:
        logger.error(f"Error retrieving answer for {subject}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
