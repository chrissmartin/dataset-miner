import argparse
import logging
from colorama import Fore, Style, init
from dotenv import load_dotenv
from cost_analyzer import CostAnalyzer
from file_processor import start_mining
from llm_utils import initialize_llm
from logging_utils import setup_logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def parse_arguments():
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
    return parser.parse_args()


def print_summaryV1(mined_data, cost_analyzer, output_file):
    logger.info(
        f"{Fore.GREEN}✨ Dataset mining complete. {len(mined_data)} Q&A pairs extracted and saved to {output_file}"
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


def print_summary(mined_data, cost_analyzer, output_file, verification_enabled):
    summary = cost_analyzer.get_summary()

    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}📊 Dataset Mining Summary 📊".center(60))
    print(f"{Fore.CYAN}{'=' * 60}\n")

    print(f"{Fore.GREEN}✨ Dataset mining complete!")
    print(f"   📁 {len(mined_data)} Q&A pairs extracted and saved to {output_file}\n")

    print(f"{Fore.MAGENTA}🔢 Token Statistics:")
    print(f"   📥 Input tokens:  {Fore.CYAN}{summary['total_input_tokens']:,}")
    print(f"   📤 Output tokens: {Fore.CYAN}{summary['total_output_tokens']:,}")
    print(f"   🔢 Total tokens:  {Fore.CYAN}{summary['total_tokens']:,}\n")

    print(f"{Fore.YELLOW}💰 Cost Breakdown:")
    print(f"   📥 Input cost:  {Fore.GREEN}${summary['total_input_cost']:,.6f}")
    print(f"   📤 Output cost: {Fore.GREEN}${summary['total_output_cost']:,.6f}")
    print(f"   💎 Total cost:  {Fore.GREEN}${summary['total_cost']:,.6f}\n")

    if len(mined_data) > 0:
        average_cost_per_pair = summary["total_cost"] / len(mined_data)
        print(
            f"{Fore.BLUE}📌 Average cost per Q&A pair: {Fore.GREEN}${average_cost_per_pair:.6f}\n"
        )

    if verification_enabled:
        print(f"\n{Fore.YELLOW}🔍 Verification Statistics:")
        print(
            f"   🔢 Verification tokens: {Fore.CYAN}{summary['total_verification_tokens']:,}"
        )
        print(
            f"   💰 Verification cost:   {Fore.GREEN}${summary['total_verification_cost']:,.6f}"
        )
    print(f"\n{Fore.YELLOW}💰 Grand Total Cost:")
    print(
        f"   💎 Total (incl. verification): {Fore.GREEN}${summary['grand_total_cost']:,.6f}\n"
    )

    print(f"{Fore.CYAN}{'=' * 60}")
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}🎉 Mining Process Completed Successfully! 🎉".center(
            60
        )
    )
    print(f"{Fore.CYAN}{'=' * 60}\n")


def main():
    init(autoreset=True)
    args = parse_arguments()
    setup_logging(args.debug)

    if args.debug:
        logger.debug("🐞 Debug logging enabled")

    logger.info(f"🚀 Dataset Miner starting with model: {args.model}")

    llm, rate_limiter = initialize_llm(args)
    if not llm:
        logger.error("❌ Failed to initialize LLM. Exiting...")
        return

    cost_analyzer = CostAnalyzer(
        model_name="gpt-4o-mini",
        input_price_per_1m_tokens=0.150,
        output_price_per_1m_tokens=0.600,
    )
    logger.info("💰 Cost analyzer initialized")

    try:
        logger.info("🏁 Starting mining process...")
        mined_data = start_mining(args, llm, cost_analyzer, rate_limiter)
        logger.info("✅ Mining process completed successfully")
    except KeyboardInterrupt:
        logger.warning("⚠️ Interrupt received. Saving mined data and exiting...")
    except Exception as e:
        logger.error(f"❌ An error occurred during mining: {str(e)}")
    finally:
        print_summary(mined_data, cost_analyzer, args.output, args.verify)


if __name__ == "__main__":
    main()
