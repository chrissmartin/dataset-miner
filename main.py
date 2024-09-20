import argparse
import json
import os
import logging
import re
from typing import List
from tqdm import tqdm
import PyPDF2
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
import docx
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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


def extract_json_from_response(response: str) -> List[dict]:
    # Find all JSON objects within the response
    json_matches = re.findall(r"\[[\s\S]*\]", response, re.DOTALL)
    results = []
    for json_str in json_matches:
        try:
            json_obj = json.loads(json_str)
            if isinstance(json_obj, list):
                results.extend(json_obj)
            elif isinstance(json_obj, dict):
                results.append(json_obj)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {json_str}")
    return results


def generate_questions_answers(text_chunk: str, llm: Ollama) -> List[dict]:
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Generate 1 question and their answer based on the following text. Respond with a JSON array containing objects with 'question' and 'answer' keys. Text: {text}",
    )
    chain = prompt | llm | StrOutputParser()

    try:
        logger.debug(f"Sending chunk of {len(text_chunk)} characters to Ollama")
        response = chain.invoke({"text": text_chunk})
        logger.debug(f"Received response of {len(response)} characters from Ollama")
        logger.debug(f"LLM Response: {response}")

        # Extract and parse JSON from the response
        result = extract_json_from_response(response)
        if result:
            logger.info(f"Generated {len(result)} Q&A pairs from this chunk")
            return result
        else:
            logger.error("No valid JSON objects found in the response")
            return []
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return []


def process_text(text: str, llm: Ollama, chunk_size: int = 2000) -> List[dict]:
    text_chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    logger.info(f"Processing {len(text_chunks)} chunks of text")
    all_responses = []
    for chunk in tqdm(text_chunks, desc="Processing chunks", unit="chunk"):
        responses = generate_questions_answers(chunk, llm)
        all_responses.extend(responses)
    logger.info(f"Generated {len(all_responses)} Q&A pairs in total")
    return all_responses


def process_file(file_path: str, llm: Ollama) -> List[dict]:
    file_extension = os.path.splitext(file_path.lower())[1]
    if file_extension == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif file_extension == ".txt":
        text = extract_text_from_txt(file_path)
    elif file_extension == ".docx":
        text = extract_text_from_docx(file_path)
    elif file_extension == ".json":
        text = extract_text_from_json(file_path)
    elif file_extension == ".csv":
        text = extract_text_from_csv(file_path)
    elif file_extension in [".xlsx", ".xls"]:
        text = extract_text_from_xlsx(file_path)
    else:
        logger.error(f"Unsupported file type: {file_path}")
        return []
    return process_text(text, llm)


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Miner: Generate Q&A pairs from various file types using AI models"
    )
    parser.add_argument(
        "-source", required=True, help="Directory containing files to mine"
    )
    parser.add_argument(
        "-model", required=True, help="AI model ID/slug to use for mining"
    )
    parser.add_argument(
        "--output",
        default="mined_dataset.json",
        help="Output JSON file for the mined dataset (default: mined_dataset.json)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Dataset Miner starting with model: {args.model}")
    llm = Ollama(model=args.model)

    all_responses = []
    for filename in os.listdir(args.source):
        if filename.lower().endswith(
            (".pdf", ".txt", ".docx", ".json", ".csv", ".xlsx", ".xls")
        ):
            file_path = os.path.join(args.source, filename)
            logger.info(f"Mining data from {filename}...")
            responses = process_file(file_path, llm)
            all_responses.extend(responses)

    with open(args.output, "w") as f:
        json.dump({"mined_data": all_responses}, f, indent=2)

    logger.info(
        f"Dataset mining complete. {len(all_responses)} Q&A pairs extracted and saved to {args.output}"
    )


if __name__ == "__main__":
    main()
