import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
)
from langchain_community.document_loaders import UnstructuredHTMLLoader

from dataset_miner.cost_analyzer import CostAnalyzer
from dataset_miner.data_extractor import extract_text
from dataset_miner.file_types import (
    EXTENSION_TO_LANGUAGE,
    get_file_extension,
    get_file_list,
    is_supported_code_file,
)
from dataset_miner.llm_utils import process_text, format_alpaca_dataset
from dataset_miner.project_types import ChatModel, CliArgs
from dataset_miner.rate_limiter import RateLimiter
from dataset_miner.verification import verify_dataset, QAPair, VerifiedQAPair

import json
import os

logger = logging.getLogger(__name__)


def start_mining(
    args: CliArgs,
    llm: ChatModel,
    cost_analyzer: CostAnalyzer,
    rate_limiter: Optional[RateLimiter] = None,
) -> Tuple[List[Union[QAPair, VerifiedQAPair]], str]:
    mined_data: List[Union[QAPair, VerifiedQAPair]] = []
    files = get_file_list(args.source)

    # Generate a unique output filename at the start
    unique_output_file = generate_unique_filename(args.output)

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
                output_file=unique_output_file,
                chunk_size=2000,
                chunk_overlap=200,
                verify=args.verify,
            )
            mined_data.extend(file_mined_data)
            logger.info(
                f"✅ Completed mining {filename}. Generated {len(file_mined_data)} Q&A pairs."
            )

    logger.info(
        f"🎉 Mining process complete. Total Q&A pairs generated: {len(mined_data)}"
    )
    return mined_data, unique_output_file


def get_appropriate_splitter(file_path: str, chunk_size: int, chunk_overlap: int):
    file_extension = get_file_extension(file_path)

    if file_extension == ".html":
        return HTMLHeaderTextSplitter(
            headers_to_split_on=[
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
                ("h5", "Header 5"),
                ("h6", "Header 6"),
                ("article", "Article"),
                ("section", "Section"),
                ("main", "Main Content"),
                ("nav", "Navigation"),
            ],
        )
    elif file_extension == ".md":
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
    elif is_supported_code_file(file_extension):
        return RecursiveCharacterTextSplitter.from_language(
            language=EXTENSION_TO_LANGUAGE[file_extension],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        logger.warning(
            f"Unsupported file extension: {file_extension}. Using default splitter."
        )
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )


def process_file(
    file_path: str,
    llm: ChatModel,
    cost_analyzer: CostAnalyzer,
    rate_limiter: Optional[RateLimiter] = None,
    output_file: str = "mined_dataset.json",
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    verify: bool = False,
) -> List[Union[QAPair, VerifiedQAPair]]:
    _, file_extension = os.path.splitext(file_path.lower())

    logger.info(f"🔧 Processing file: {file_path}")
    if file_extension == ".html":
        loader = UnstructuredHTMLLoader(file_path)
        data = loader.load()
        text = data[0].page_content
        logger.info(f"📄 Loaded HTML content: {len(text)} characters")
    else:
        text = extract_text(file_path)
        if not text:
            logger.warning(
                f"⚠️ No text extracted from {file_path}. Skipping processing."
            )
            return []
        logger.info(f"📄 Extracted text: {len(text)} characters")

    mined_data: List[Union[QAPair, VerifiedQAPair]] = []
    if text:
        text_splitter = get_appropriate_splitter(file_path, chunk_size, chunk_overlap)
        text_chunks = text_splitter.split_text(text)

        logger.info(f"🧩 Splitting text into {len(text_chunks)} chunks")

        for i, chunk in enumerate(
            tqdm(text_chunks, desc="🔍 Processing chunks", unit="chunk", leave=False)
        ):
            chunk_context = (
                chunk.page_content if hasattr(chunk, "page_content") else chunk
            )
            qa_pairs: List[QAPair] = process_text(
                chunk_context, llm, cost_analyzer, rate_limiter
            )

            if verify:
                verified_chunk_data: List[VerifiedQAPair] = verify_dataset(
                    chunk_context, qa_pairs, llm, cost_analyzer, rate_limiter
                )
                mined_data.extend(verified_chunk_data)
                logger.info(
                    f"✅ Chunk {i+1}/{len(text_chunks)} processed and verified. Generated {len(verified_chunk_data)} verified Q&A pairs."
                )
            else:
                mined_data.extend(qa_pairs)
                logger.info(
                    f"✅ Chunk {i+1}/{len(text_chunks)} processed. Generated {len(qa_pairs)} Q&A pairs."
                )

        # Save after each chunk
        save_mined_data(mined_data, output_file)
        logger.info(f"💾 Saved {len(mined_data)} Q&A pairs to {output_file}")
    return mined_data


def generate_unique_filename(base_filename: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory, filename = os.path.split(base_filename)
    name, ext = os.path.splitext(filename)
    return os.path.join(directory, f"{name}_{timestamp}{ext}")


def save_mined_data(
    mined_data: List[Union[QAPair, VerifiedQAPair]], output_file_path: str
) -> None:
    # Convert Union type to Dict before passing to format_alpaca_dataset
    data_as_dicts: List[Dict[str, Any]] = [dict(qa_pair) for qa_pair in mined_data]
    formatted_data = format_alpaca_dataset(data_as_dicts)

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path) or ".", exist_ok=True)

    # Check if the file exists and has content
    if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
        # If file exists, read existing data
        with open(output_file_path, "r") as f:
            existing_data = json.load(f)

        # Append new data to existing data
        existing_data.extend(formatted_data)

        # Write combined data back to file
        with open(output_file_path, "w") as f:
            json.dump(existing_data, f, indent=2)

        logger.info(
            f"Appended {len(formatted_data)} Q&A pairs to {output_file_path}. Total pairs: {len(existing_data)}"
        )
    else:
        # If file doesn't exist or is empty, write new data
        with open(output_file_path, "w") as f:
            json.dump(formatted_data, f, indent=2)

        logger.info(f"Saved {len(formatted_data)} Q&A pairs to {output_file_path}")
