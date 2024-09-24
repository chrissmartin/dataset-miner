import logging
from typing import List, Dict
from tqdm import tqdm
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
)
from langchain_community.document_loaders import UnstructuredHTMLLoader
from cost_analyzer import CostAnalyzer
from data_extractor import extract_text
from llm_utils import process_text, format_alpaca_dataset, RateLimiter
from verification import verify_dataset
import json
import os

logger = logging.getLogger(__name__)


def get_file_list(source_dir):
    return [
        f
        for f in os.listdir(source_dir)
        if f.lower().endswith(
            (".pdf", ".txt", ".docx", ".json", ".csv", ".xlsx", ".xls")
        )
    ]


def start_mining(args, llm, cost_analyzer, rate_limiter):
    mined_data = []
    files = get_file_list(args.source)

    logger.info(f"🔍 Starting mining process on {len(files)} files")
    with tqdm(files, desc="📂 Processing files", unit="file") as pbar:
        for filename in pbar:
            file_path = os.path.join(args.source, filename)
            pbar.set_description(f"📄 Processing {filename}")
            logger.info(f"📄 Mining data from {filename}...")
            file_mined_data = process_file(
                file_path,
                llm,
                cost_analyzer,
                rate_limiter,
                remove_empty=args.remove_empty_columns,
                output_file=args.output,
                chunk_size=2000,
                chunk_overlap=200,
            )
            mined_data.extend(file_mined_data)
            logger.info(
                f"✅ Completed mining {filename}. Generated {len(file_mined_data)} Q&A pairs."
            )

    logger.info(
        f"🎉 Mining process complete. Total Q&A pairs generated: {len(mined_data)}"
    )
    return mined_data


def get_appropriate_splitter(file_path: str, chunk_size: int, chunk_overlap: int):
    _, file_extension = os.path.splitext(file_path.lower())

    if file_extension == ".html":
        return HTMLHeaderTextSplitter(
            tags=["h1", "h2", "h3", "h4", "h5", "h6"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif file_extension == ".md":
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
    elif file_extension in [".py", ".java", ".cpp", ".js", ".ts"]:
        return RecursiveCharacterTextSplitter.from_language(
            language=file_extension[1:],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )


def process_file(
    file_path: str,
    llm,
    cost_analyzer: CostAnalyzer,
    rate_limiter: RateLimiter = None,
    remove_empty: bool = False,
    output_file: str = "mined_dataset.json",
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> List[Dict]:
    _, file_extension = os.path.splitext(file_path.lower())

    logger.info(f"🔧 Processing file: {file_path}")
    if file_extension == ".html":
        loader = UnstructuredHTMLLoader(file_path)
        data = loader.load()
        text = data[0].page_content
        logger.info(f"📄 Loaded HTML content: {len(text)} characters")
    else:
        text = extract_text(file_path, remove_empty)
        logger.info(f"📄 Extracted text: {len(text)} characters")

    mined_data = []
    if text:
        text_splitter = get_appropriate_splitter(file_path, chunk_size, chunk_overlap)
        text_chunks = (
            text_splitter.split_text(text)
            if not isinstance(text_splitter, HTMLHeaderTextSplitter)
            else text_splitter.split_text(text)
        )

        logger.info(f"🧩 Splitting text into {len(text_chunks)} chunks")

        for i, chunk in enumerate(
            tqdm(text_chunks, desc="🔍 Processing chunks", unit="chunk", leave=False)
        ):
            chunk_text = chunk.page_content if hasattr(chunk, "page_content") else chunk
            chunk_data = process_text(chunk_text, llm, cost_analyzer, rate_limiter)
            verified_chunk_data = verify_dataset(
                chunk_text, chunk_data, llm, cost_analyzer, rate_limiter
            )
            mined_data.extend(verified_chunk_data)
            logger.info(
                f"✅ Chunk {i+1}/{len(text_chunks)} processed. Generated {len(verified_chunk_data)} verified Q&A pairs."
            )

            save_mined_data(mined_data, output_file)
            logger.info(f"💾 Saved {len(mined_data)} Q&A pairs to {output_file}")

    return mined_data


def save_mined_data(mined_data: List[Dict], output_file_path: str):
    formatted_data = format_alpaca_dataset(mined_data)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    with open(output_file_path, "w") as f:
        json.dump(formatted_data, f, indent=2)
    logger.info(f"Saved {len(formatted_data)} Q&A pairs to {output_file_path}")
