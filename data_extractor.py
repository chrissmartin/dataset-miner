import logging
import PyPDF2
import docx
import pandas as pd
import json

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    logger.info(f"Extracting text from PDF: {file_path}")
    with open(file_path, "rb") as pdf_file_obj:
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    logger.info(f"Extracted {len(text)} characters from {file_path}")
    return text


def extract_text_from_txt(file_path: str) -> str:
    logger.info(f"Extracting text from TXT: {file_path}")
    with open(file_path, "r", encoding="utf-8") as txt_file:
        text = txt_file.read()
    logger.info(f"Extracted {len(text)} characters from {file_path}")
    return text


def extract_text_from_docx(file_path: str) -> str:
    logger.info(f"Extracting text from DOCX: {file_path}")
    doc = docx.Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    logger.info(f"Extracted {len(text)} characters from {file_path}")
    return text


def extract_text_from_json(file_path: str) -> str:
    logger.info(f"Extracting text from JSON: {file_path}")
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    text = json.dumps(data, indent=2)
    logger.info(f"Extracted {len(text)} characters from {file_path}")
    return text


def extract_text_from_csv(file_path: str) -> str:
    logger.info(f"Extracting text from CSV: {file_path}")
    df = pd.read_csv(file_path)
    text = df.to_string(index=False)
    logger.info(f"Extracted {len(text)} characters from {file_path}")
    return text


def extract_text_from_xlsx(file_path: str) -> str:
    logger.info(f"Extracting text from XLSX: {file_path}")
    df = pd.read_excel(file_path, sheet_name=None)
    text = ""
    for sheet_name, sheet_data in df.items():
        text += f"Sheet: {sheet_name}\n"
        text += sheet_data.to_string(index=False)
        text += "\n\n"
    logger.info(f"Extracted {len(text)} characters from {file_path}")
    return text


def extract_text(file_path: str) -> str:
    file_extension = file_path.lower().split(".")[-1]
    extraction_functions = {
        "pdf": extract_text_from_pdf,
        "txt": extract_text_from_txt,
        "docx": extract_text_from_docx,
        "json": extract_text_from_json,
        "csv": extract_text_from_csv,
        "xlsx": extract_text_from_xlsx,
        "xls": extract_text_from_xlsx,
    }
    if file_extension in extraction_functions:
        return extraction_functions[file_extension](file_path)
    else:
        logger.error(f"Unsupported file type: {file_path}")
        return ""
