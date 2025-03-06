# Extract Text Functions
from docx import Document
from pptx import Presentation
from .Simple_preprocess import preprocess_text
import PyPDF2


def extract_text_from_pdf(file_path):
    extracted_text = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
    return " \n ".join(extracted_text)


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ". \n ".join([p.text for p in doc.paragraphs if p.text.strip()])
    return preprocess_text(text)


def extract_text_from_ppt(file_path):
    text = ""
    ppt = Presentation(file_path)

    for slide in ppt.slides:
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame is not None:  # Ensure text_frame exists
                for paragraph in shape.text_frame.paragraphs:
                    for run in paragraph.runs:
                        text += run.text + " "  # Preserve spacing between words
                    text += ". \n "  # Separate paragraphs with a newline

    return preprocess_text(text.strip())  # Remove unnecessary trailing spaces
