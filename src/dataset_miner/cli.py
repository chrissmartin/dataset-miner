import argparse
import logging
from colorama import init
from dotenv import load_dotenv

from dataset_miner.cost_analyzer import CostAnalyzer
from dataset_miner.file_processor import start_mining
from dataset_miner.llm_utils import initialize_llm
from dataset_miner.logging_utils import setup_logging
from dataset_miner.project_types import CliArgs
from dataset_miner.summary_log import print_summary
from dataset_miner import __version__
from dataset_miner.verification import QAPair, VerifiedQAPair

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def parse_arguments() -> CliArgs:
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
        help="Output JSON file for the mined dataset",
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable verification of generated Q&A pairs",
    )
    args = parser.parse_args()
    return CliArgs(
        source=args.source,
        model=args.model,
        output=args.output,
        debug=args.debug,
        use_groq=args.use_groq,
        remove_empty_columns=args.remove_empty_columns,
        verify=args.verify,
    )


def main() -> None:
    """Main entry point for the dataset-miner CLI."""
    print(f"Dataset Miner version: {__version__}")
    init(autoreset=True)
    args: CliArgs = parse_arguments()
    setup_logging(args.debug)
    mined_data: list[QAPair | VerifiedQAPair] = []
    output_file_path = ""

    if args.debug:
        logger.debug("🐞 Debug logging enabled")

    try:
        logger.info(f"🚀 Dataset Miner starting with model: {args.model}")
        llm, rate_limiter = initialize_llm(args)
        if not llm:
            logger.error("❌ Failed to initialize LLM. Exiting...")
            return

        logger.info("💰 Initializing cost analyzer")
        cost_analyzer = CostAnalyzer(
            model_name="gpt-4o-mini",
            input_price_per_1m_tokens=0.150,
            output_price_per_1m_tokens=0.600,
        )
        logger.info("💰 Cost analyzer initialized")
        logger.info("🏁 Starting mining process...")
        mined_data, output_file_path = start_mining(
            args, llm, cost_analyzer, rate_limiter
        )
        logger.info("✅ Mining process completed successfully")
    except KeyboardInterrupt:
        logger.warning("⚠️ Interrupt received. Saving mined data and exiting...")
    except Exception as e:
        logger.error(f"❌ An error occurred during mining: {str(e)}")
    finally:
        print_summary(mined_data, cost_analyzer, output_file_path, args.verify)


if __name__ == "__main__":
    main()
