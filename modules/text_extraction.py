import os
import PyPDF2
import logging
from docx import Document
from pptx import Presentation
import fitz  # PyMuPDF
from typing import List, Optional, Union
from .Simple_preprocess import preprocess_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF files using PyMuPDF for better extraction quality.
    Falls back to PyPDF2 if PyMuPDF fails.
    """
    try:
        # Trying PyMuPDF first - better for complex PDFs
        doc = fitz.open(file_path)
        extracted_text: List[str] = []

        metadata = doc.metadata
        if metadata and isinstance(metadata, dict) and metadata.get('title'):
            extracted_text.append(f"Document Title: {metadata.get('title')}")

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                extracted_text.append(text)

        return preprocess_text("\n".join(extracted_text))

    except Exception as e:
        logger.warning(
            f"PyMuPDF extraction failed for {file_path}, falling back to PyPDF2: {e}")

        # Fallback to PyPDF2
        try:
            extracted_text: List[str] = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)

                if reader.metadata and hasattr(reader.metadata, 'title') and reader.metadata.title:
                    extracted_text.append(
                        f"Document Title: {reader.metadata.title}")

                # Extracting text from each page
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text.append(text)

            return preprocess_text("\n".join(extracted_text))

        except Exception as e:
            logger.error(f"Failed to extract text from PDF {file_path}: {e}")
            return f"Error extracting text from {os.path.basename(file_path)}"


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from DOCX files with improved structure preservation.
    Includes tables, headers, and other document elements.
    """
    try:
        doc = Document(file_path)
        content: List[str] = []

        prop = doc.core_properties
        if hasattr(prop, 'title') and prop.title:
            content.append(f"Document Title: {prop.title}")

        # Processing paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                if para.style and hasattr(para.style, 'name') and para.style.name and \
                   isinstance(para.style.name, str) and para.style.name.startswith('Heading'):
                    # Adding formatting to preserve heading structure
                    content.append(f"\n## {para.text} ##\n")
                else:
                    content.append(para.text)

        # Processing tables
        for table in doc.tables:
            table_text: List[str] = []
            for row in table.rows:
                row_text = [cell.text.strip()
                            for cell in row.cells if cell.text.strip()]
                if row_text:
                    table_text.append(" | ".join(row_text))
            if table_text:
                content.append("\nTable Content:\n" + "\n".join(table_text))

        return preprocess_text("\n".join(content))

    except Exception as e:
        logger.error(f"Failed to extract text from DOCX {file_path}: {e}")
        return f"Error extracting text from {os.path.basename(file_path)}"


def extract_text_from_ppt(file_path: str) -> str:
    """
    Extract text from PowerPoint presentations with improved structure.
    Includes slide numbers, titles, and notes.
    """
    try:
        ppt = Presentation(file_path)
        content: List[str] = []

        if hasattr(ppt.core_properties, 'title') and ppt.core_properties.title:
            content.append(f"Presentation Title: {ppt.core_properties.title}")

        # Processing slides
        for slide_num, slide in enumerate(ppt.slides, 1):
            slide_content: List[str] = [f"\nSlide {slide_num}:"]

            if slide.shapes.title and hasattr(slide.shapes.title, 'text') and slide.shapes.title.text:
                slide_content.append(f"Title: {slide.shapes.title.text}")

            shape_texts: List[str] = []

            # Processing shapes
            for shape in slide.shapes:
                # Safe check for text frame using hasattr
                if not hasattr(shape, 'has_text_frame') or not shape.has_text_frame:
                    continue

                # Safe check for text_frame attribute
                if not hasattr(shape, 'text_frame'):
                    continue

                # Extract text from the shape
                shape_text: List[str] = []
                for paragraph in shape.text_frame.paragraphs:
                    paragraph_text = " ".join(
                        run.text for run in paragraph.runs if hasattr(run, 'text'))
                    if paragraph_text.strip():
                        shape_text.append(paragraph_text)

                if shape_text:
                    shape_texts.append("\n".join(shape_text))

            if shape_texts:
                slide_content.append("Content: " + "\n".join(shape_texts))

            # Extracting notes if available
            if hasattr(slide, 'has_notes_slide') and slide.has_notes_slide and \
               hasattr(slide, 'notes_slide') and slide.notes_slide and \
               hasattr(slide.notes_slide, 'notes_text_frame'):
                notes: List[str] = []
                for paragraph in slide.notes_slide.notes_text_frame.paragraphs:
                    paragraph_text = " ".join(
                        run.text for run in paragraph.runs if hasattr(run, 'text'))
                    if paragraph_text.strip():
                        notes.append(paragraph_text)

                if notes:
                    slide_content.append("Notes: " + "\n".join(notes))

            content.append("\n".join(slide_content))

        return preprocess_text("\n".join(content))

    except Exception as e:
        logger.error(f"Failed to extract text from PPT {file_path}: {e}")
        return f"Error extracting text from {os.path.basename(file_path)}"
