import os
import logging
from typing import List, Dict, Optional, Callable, Any
from sentence_transformers import SentenceTransformer
from .subject_processor import (
    get_available_subjects,
    process_subject_knowledge_base,
    get_answer_for_subject
)

logger = logging.getLogger(__name__)


class EduQueryCore:
    def __init__(self, model_path: str, data_folder: str, indices_folder: str):
        """Initialize the core application logic"""
        self.model_path = model_path
        self.data_folder = data_folder
        self.indices_folder = indices_folder
        self.model = None

        logger.info(
            f"Initializing EduQueryCore with data_folder: {data_folder}")

        # Ensure folders exist
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(indices_folder, exist_ok=True)

    def load_model(self) -> bool:
        """Load the sentence transformer model"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model path not found: {self.model_path}")
                return False
            self.model = SentenceTransformer(self.model_path)
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def ensure_indices_folder(self) -> None:
        """Create indices folder if it doesn't exist"""
        os.makedirs(self.indices_folder, exist_ok=True)

    def get_subjects(self) -> List[str]:
        """
        Dynamically get list of available subjects from data folder.
        Returns empty list if no subjects found.
        """
        return get_available_subjects(self.data_folder)

    def check_missing_indices(self) -> List[str]:
        """Check which subjects have missing indices"""
        subjects = self.get_subjects()
        missing_indices = []

        for subject in subjects:
            subject_indices_folder = os.path.join(self.indices_folder, subject)
            faiss_index_file = os.path.join(
                subject_indices_folder, "faiss_index.idx")
            if not os.path.exists(faiss_index_file):
                missing_indices.append(subject)

        return missing_indices

    def initialize_subject(self, subject: str) -> bool:
        """Initialize knowledge base for a specific subject"""
        if not self.model:
            if not self.load_model():
                return False

        return process_subject_knowledge_base(
            self.data_folder,
            self.indices_folder,
            subject,
            self.model
        )

    def initialize_knowledge_bases(self, progress_callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """Initialize knowledge bases for all subjects"""
        subjects = self.get_subjects()

        if not subjects:
            logger.warning("No subject folders found in data directory")
            return False

        success_count = 0
        for i, subject in enumerate(subjects):
            success = process_subject_knowledge_base(
                self.data_folder,
                self.indices_folder,
                subject,
                self.model
            )

            if success:
                success_count += 1

            if progress_callback:
                progress_callback(subject, (i + 1) / len(subjects))

        return success_count > 0

    def check_subject_index(self, subject: str) -> bool:
        """Check if a specific subject's index exists"""
        subject_indices_folder = os.path.join(self.indices_folder, subject)
        faiss_index_file = os.path.join(
            subject_indices_folder, "faiss_index.idx")
        return os.path.exists(faiss_index_file)

    def get_answer(self, query: str, subject: str) -> Optional[List[Dict[str, Any]]]:
        """Get answer for a specific query and subject as structured results."""
        if not self.model:
            logger.error("Model not loaded")
            return None

        if not self.check_subject_index(subject):
            logger.warning(f"Subject index not found for {subject}")
            return None

        return get_answer_for_subject(
            query.lower(), subject, self.indices_folder, self.model)
