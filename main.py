import argparse
import json
import os
import logging
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from cost_analyzer import CostAnalyzer
from data_extractor import extract_text
from llm_utils import (
    process_text,
    format_alpaca_dataset,
    RateLimiter,
    GROQ_REQUESTS_PER_MINUTE,
    GROQ_TOKENS_PER_MINUTE,
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variable to store mined data
mined_data = []


# Function to save mined data
def save_mined_data(output_file):
    formatted_data = format_alpaca_dataset(mined_data)
    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=2)
    logger.info(f"Saved {len(formatted_data)} Q&A pairs to {output_file}")


def process_file(
    file_path: str,
    llm,
    cost_analyzer: CostAnalyzer,
    rate_limiter: RateLimiter = None,
    remove_empty: bool = False,
) -> None:
    text = extract_text(file_path, remove_empty)
    if text:
        global mined_data
        chunk_size = 2000
        text_chunks = [
            text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
        ]
        logger.info(f"Processing file in {len(text_chunks)} chunks")

        for i, chunk in enumerate(
            tqdm(text_chunks, desc="Processing chunks", leave=False)
        ):
            chunk_data = process_text(chunk, llm, cost_analyzer, rate_limiter)
            mined_data.extend(chunk_data)
            logger.info(
                f"Chunk {i+1}/{len(text_chunks)} processed. Generated {len(chunk_data)} Q&A pairs."
            )

            # Save after each chunk
            save_mined_data(args.output)


def main():
    global args
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
    parser.add_argument(
        "--use-groq", action="store_true", help="Use Groq instead of Ollama"
    )
    parser.add_argument(
        "--remove-empty-columns",
        action="store_true",
        help="Remove empty columns from CSV and Excel files",
    )
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info(f"Dataset Miner starting with model: {args.model}")

    if args.use_groq:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.error(
                "GROQ_API_KEY not found in environment variables. Please set it in your .env file."
            )
            return
        llm = ChatGroq(model_name=args.model, groq_api_key=groq_api_key)
        rate_limiter = RateLimiter(GROQ_REQUESTS_PER_MINUTE, GROQ_TOKENS_PER_MINUTE)
        logger.info("Using Groq with rate limiting")
    else:
        llm = Ollama(model=args.model)
        rate_limiter = None
        logger.info("Using Ollama")

    cost_analyzer = CostAnalyzer(
        model_name="gpt-4o-mini",
        input_price_per_1m_tokens=0.150,
        output_price_per_1m_tokens=0.600,
    )

    try:
        files = [
            f
            for f in os.listdir(args.source)
            if f.lower().endswith(
                (".pdf", ".txt", ".docx", ".json", ".csv", ".xlsx", ".xls")
            )
        ]

        with tqdm(files, desc="Processing files") as pbar:
            for filename in pbar:
                file_path = os.path.join(args.source, filename)
                pbar.set_description(f"Processing {filename}")
                logger.info(f"Mining data from {filename}...")
                process_file(
                    file_path,
                    llm,
                    cost_analyzer,
                    rate_limiter,
                    args.remove_empty_columns,
                )

    except KeyboardInterrupt:
        logger.info("Interrupt received. Saving mined data and exiting...")
    finally:
        # Final save
        save_mined_data(args.output)

        logger.info(
            f"Dataset mining complete. {len(mined_data)} Q&A pairs extracted and saved to {args.output}"
        )

        summary = cost_analyzer.get_summary()
        logger.info(f"Total input tokens processed: {summary['total_input_tokens']}")
        logger.info(f"Total output tokens generated: {summary['total_output_tokens']}")
        logger.info(f"Total tokens: {summary['total_tokens']}")
        logger.info(f"Total input cost: ${summary['total_input_cost']:.6f}")
        logger.info(f"Total output cost: ${summary['total_output_cost']:.6f}")
        logger.info(f"Total estimated cost: ${summary['total_cost']:.6f}")
        if len(mined_data) > 0:
            average_cost_per_pair = summary["total_cost"] / len(mined_data)
            logger.info(f"Average cost per Q&A pair: ${average_cost_per_pair:.6f}")


if __name__ == "__main__":
    main()
