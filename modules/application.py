import os
import logging
from typing import List, Dict, Optional, Callable, Any
from sentence_transformers import SentenceTransformer
from .subject_processor import (
    get_available_subjects,
    process_subject_knowledge_base,
    get_answer_for_subject
)
from .synthesizer import synthesize_answer_with_llm

logger = logging.getLogger(__name__)


class EduQueryCore:
    def __init__(self, model_path: str, data_folder: str, indices_folder: str):
        self.model_path = model_path
        self.data_folder = data_folder
        self.indices_folder = indices_folder
        self.model: Optional[SentenceTransformer] = None
        self.subjects: List[str] = []
        self.ensure_indices_folder()
        self.subjects = get_available_subjects(self.data_folder)

    def ensure_indices_folder(self):
        os.makedirs(self.indices_folder, exist_ok=True)

    def load_model(self) -> bool:
        """Loads the sentence transformer model onto the CPU."""
        if self.model:
            logger.info("Embedding model already loaded.")
            return True
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model path does not exist: {self.model_path}")
                return False
            logger.info(f"Loading embedding model from: {self.model_path}")
            self.model = SentenceTransformer(self.model_path, device='cpu')
            logger.info("Embedding model loaded successfully onto CPU.")
            return True
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
            return False

    def unload_model(self):
        """Unloads the sentence transformer model."""
        if self.model:
            logger.info("Unloading embedding model.")
            del self.model
            self.model = None
            logger.info("Embedding model unloaded.")
        else:
            logger.info("Embedding model not loaded, nothing to unload.")

    def get_subjects(self) -> List[str]:
        """Returns the list of available subjects."""
        self.subjects = get_available_subjects(self.data_folder)
        return self.subjects

    def check_missing_indices(self) -> List[str]:
        """Checks which subjects are missing their index files."""
        missing = []
        for subject in self.subjects:
            if not self.check_subject_index(subject):
                missing.append(subject)
        return missing

    def check_subject_index(self, subject: str) -> bool:
        """Checks if the necessary index files exist for a subject."""
        subject_indices_folder = os.path.join(self.indices_folder, subject)
        faiss_index_file = os.path.join(
            subject_indices_folder, "faiss_index.idx")
        chunks_file = os.path.join(subject_indices_folder, "chunks.json")
        # bm25_file = os.path.join(subject_indices_folder, "bm25.json") # BM25 check optional
        return os.path.exists(faiss_index_file) and os.path.exists(chunks_file)

    def initialize_knowledge_bases(self, progress_callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """Initializes knowledge bases for all subjects."""
        all_successful = True
        # --- Load model before processing ---
        if not self.load_model():
            logger.error(
                "Cannot initialize knowledge bases: Failed to load embedding model.")
            return False

        try:
            total_subjects = len(self.subjects)
            for i, subject in enumerate(self.subjects):
                logger.info(f"Processing subject: {subject}")
                if not process_subject_knowledge_base(self.data_folder, self.indices_folder, subject, self.model):
                    logger.error(
                        f"Failed to process knowledge base for {subject}")
                    all_successful = False
                if progress_callback:
                    progress_callback(subject, (i + 1) / total_subjects)
            return all_successful
        finally:
            # --- Unload model after processing ---
            self.unload_model()

    def initialize_subject(self, subject: str) -> bool:
        """Initializes the knowledge base for a single subject."""
        # --- Load model before processing ---
        if not self.load_model():
            logger.error(
                f"Cannot initialize subject '{subject}': Failed to load embedding model.")
            return False

        try:
            if subject not in self.subjects:
                logger.error(f"Subject '{subject}' not found in data folder.")
                return False
            logger.info(f"Initializing knowledge base for subject: {subject}")
            success = process_subject_knowledge_base(
                self.data_folder, self.indices_folder, subject, self.model)
            if success:
                logger.info(
                    f"Successfully initialized knowledge base for {subject}")
            else:
                logger.error(
                    f"Failed to initialize knowledge base for {subject}")
            return success
        finally:
            self.unload_model()

    def get_answer(self, query: str, subject: str) -> Optional[str]:
        """Retrieves relevant chunks and synthesizes a final answer using an LLM."""
        if not self.model:
            logger.info(
                "Embedding model not loaded for get_answer, loading now.")
            if not self.load_model():
                return "Error: The embedding model required for searching could not be loaded."

        if not self.check_subject_index(subject):
            logger.warning(
                f"Subject index not found or incomplete for {subject}")
            return f"Error: The knowledge base for '{subject}' has not been initialized or is incomplete."

        # 1. Retrieve relevant chunks (using the function that returns List[Dict])
        logger.info(
            f"Retrieving chunks for query: '{query}' in subject: '{subject}'")
        retrieved_chunks = get_answer_for_subject(
            query.lower(), subject, self.indices_folder, self.model
        )

        # Handle case where retrieval itself fails or returns None
        if retrieved_chunks is None:
            logger.error(f"Chunk retrieval failed for subject {subject}.")
            # Provide a more specific error if possible, otherwise generic
            return "Error: Failed to retrieve information from the knowledge base. Check logs."

        if not retrieved_chunks:
            logger.warning("No relevant chunks found by retrieval process.")
            # Return a user-friendly message indicating nothing was found
            return "Sorry, I couldn't find relevant information for your query in the knowledge base."

        # 2. Synthesize the answer using the LLM
        logger.info(
            f"Synthesizing answer from {len(retrieved_chunks)} retrieved chunks.")
        synthesized_answer = synthesize_answer_with_llm(
            query, retrieved_chunks)

        if synthesized_answer is None:
            logger.error("Answer synthesis returned None unexpectedly.")
            return "Error: Failed to generate a summarized answer (synthesis returned None)."
        elif synthesized_answer.startswith("Error:"):
            logger.error(
                f"Answer synthesis failed with message: {synthesized_answer}")
            return synthesized_answer

        logger.info("Synthesis successful.")
        return synthesized_answer


def process_query(query: str, subject: str) -> Dict[str, Any]:
    """
    Process a user query and return answer with source information.
    This function serves as a wrapper around EduQueryCore functionality
    to provide a simpler interface for the Streamlit app.

    Args:
        query: The user's question
        subject: The selected subject area

    Returns:
        A dictionary containing the answer and sources:
        {'answer': str, 'sources': Optional[Dict[str, str]]}
    """
    logger.info(f"Processing query for subject '{subject}': '{query}'")

    # Paths from environment variables or defaults
    model_path = os.environ.get("MODEL_PATH", "models/all-mpnet-base-v2")
    data_folder = os.environ.get("DATA_FOLDER", "data")
    indices_folder = os.environ.get("INDICES_FOLDER", "indices")

    # Initialize the core handler
    core = EduQueryCore(model_path, data_folder, indices_folder)

    # Check if subject index exists
    if not core.check_subject_index(subject):
        error_msg = f"Error: Knowledge base for '{subject}' has not been initialized."
        logger.error(error_msg)
        return {'answer': error_msg, 'sources': None}

    # Load the model
    if not core.load_model():
        error_msg = "Error: Failed to load the embedding model."
        logger.error(error_msg)
        return {'answer': error_msg, 'sources': None}

    try:
        # Get answer using core functionality
        answer = core.get_answer(query, subject)

        # At this point, we only have the answer text, no separate source info
        # In a future enhancement, we could modify get_answer to return sources separately

        # Since we don't have source mapping right now, return None for sources
        return {
            'answer': answer if answer else "Sorry, couldn't generate an answer.",
            'sources': None  # Future enhancement: get actual sources
        }
    finally:
        # Always unload the model after processing
        core.unload_model()
