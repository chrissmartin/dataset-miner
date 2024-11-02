from typing import List, Dict, Any
from colorama import Fore, Style

from dataset_miner.cost_analyzer import CostAnalyzer

SEPARATOR_LENGTH = 60


def format_currency(amount: float) -> str:
    """Format a currency value with 6 decimal places."""
    return f"${amount:,.6f}"


def print_header(text: str) -> None:
    """Print a centered header with decorative separators."""
    separator = f"{Fore.CYAN}{'=' * SEPARATOR_LENGTH}"
    centered_text = f"{Fore.YELLOW}{Style.BRIGHT}{text}".center(SEPARATOR_LENGTH)
    print(f"\n{separator}")
    print(centered_text)
    print(f"{separator}\n")


def print_token_stats(summary: Dict[str, Any]) -> None:
    """Print token-related statistics."""
    print(f"{Fore.MAGENTA}🔢 Token Statistics:")
    print(f"   📥 Input tokens:  {Fore.CYAN}{summary['total_input_tokens']:,}")
    print(f"   📤 Output tokens: {Fore.CYAN}{summary['total_output_tokens']:,}")
    print(f"   🔢 Total tokens:  {Fore.CYAN}{summary['total_tokens']:,}\n")


def print_cost_breakdown(summary: Dict[str, Any]) -> None:
    """Print cost breakdown information."""
    print(f"{Fore.YELLOW}💰 Cost Breakdown:")
    print(
        f"   📥 Input cost:  {Fore.GREEN}{format_currency(summary['total_input_cost'])}"
    )
    print(
        f"   📤 Output cost: {Fore.GREEN}{format_currency(summary['total_output_cost'])}"
    )
    print(f"   💎 Total cost:  {Fore.GREEN}{format_currency(summary['total_cost'])}\n")


def print_verification_stats(summary: Dict[str, Any]) -> None:
    """Print verification statistics."""
    print(f"{Fore.YELLOW}🔍 Verification Statistics:")
    print(
        f"   🔢 Verification tokens: {Fore.CYAN}{summary['total_verification_tokens']:,}"
    )
    print(
        f"   💰 Verification cost:   {Fore.GREEN}{format_currency(summary['total_verification_cost'])}"
    )


def print_summary(
    mined_data: List,
    cost_analyzer: CostAnalyzer,
    output_file_path: str,
    verification_enabled: bool,
) -> None:
    """
    Print a formatted summary of the mining process results.

    Args:
        mined_data: List of extracted Q&A pairs
        cost_analyzer: Object containing cost analysis data
        output_file_path: Path where the data was saved
        verification_enabled: Whether verification was performed
    """
    summary = cost_analyzer.get_summary()

    # Print header
    print_header("📊 Dataset Mining Summary 📊")

    # Print dataset info
    print(f"{Fore.GREEN}✨ Dataset mining complete!")
    print(
        f"   📁 {len(mined_data)} Q&A pairs extracted and saved to {output_file_path}\n"
    )

    # Print token and cost statistics
    print_token_stats(summary)
    print_cost_breakdown(summary)

    # Print average cost per pair if there's data
    if mined_data:
        average_cost = summary["total_cost"] / len(mined_data)
        print(
            f"{Fore.BLUE}📌 Average cost per Q&A pair: {Fore.GREEN}{format_currency(average_cost)}\n"
        )

    # Print verification stats if enabled
    if verification_enabled:
        print_verification_stats(summary)

    # Print grand total
    print(f"\n{Fore.YELLOW}💰 Grand Total Cost:")
    print(
        f"   💎 Total (incl. verification): {Fore.GREEN}{format_currency(summary['grand_total_cost'])}\n"
    )

    # Print footer
    print_header("🎉 Mining Process Completed Successfully! 🎉")
