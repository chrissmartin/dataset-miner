"""Dataset Miner: A tool for generating Q&A datasets from various document formats using AI models.

This package provides functionality to:
- Process multiple document formats (PDF, DOCX, TXT, CSV, XLSX, JSON)
- Generate Q&A pairs using AI models
- Verify generated Q&A pairs
- Track token usage and costs
"""

from importlib.metadata import version, PackageNotFoundError

# Core functionality imports
from dataset_miner.file_processor import start_mining
from dataset_miner.llm_utils import initialize_llm, process_text
from dataset_miner.cost_analyzer import CostAnalyzer
from dataset_miner.data_extractor import extract_text
from dataset_miner.verification import QAPair, VerifiedQAPair, verify_dataset
from dataset_miner.project_types import CliArgs, ChatModel

# Try to get the version from package metadata
try:
    __version__ = version("dataset-miner")
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed
    try:
        from dataset_miner._version import version as __version__  # type: ignore
    except ImportError:
        __version__ = "unknown"

# Public API
__all__ = [
    "start_mining",
    "initialize_llm",
    "process_text",
    "extract_text",
    "verify_dataset",
    "CostAnalyzer",
    "CliArgs",
    "ChatModel",
    "__version__",
]


def mine_documents(
    source_dir: str,
    model: str,
    *,
    output_file: str = "mined_dataset.json",
    use_groq: bool = False,
    remove_empty_columns: bool = False,
    verify: bool = False,
    debug: bool = False,
) -> tuple[list[QAPair | VerifiedQAPair], str]:
    """
    High-level function to mine documents from a directory using specified AI model.

    Args:
        source_dir: Directory containing documents to mine
        model: AI model ID/slug to use for mining
        output_file: Path to save the mined dataset
        use_groq: Whether to use Groq instead of Ollama
        remove_empty_columns: Whether to remove empty columns from CSV/Excel files
        verify: Whether to verify generated Q&A pairs
        debug: Whether to enable debug logging

    Returns:
        tuple: (List of mined data dictionaries, Output file path)

    Example:
        >>> from dataset_miner import mine_documents
        >>> mined_data, output_path = mine_documents(
        ...     source_dir="./documents",
        ...     model="gpt-4o-mini",
        ...     output_file="dataset.json",
        ...     verify=True
        ... )
    """
    args = CliArgs(
        source=source_dir,
        model=model,
        output=output_file,
        use_groq=use_groq,
        remove_empty_columns=remove_empty_columns,
        verify=verify,
        debug=debug,
    )
    llm, rate_limiter = initialize_llm(args)
    if not llm:
        raise RuntimeError("Failed to initialize LLM")

    cost_analyzer = CostAnalyzer(
        model_name="gpt-4o-mini",
        input_price_per_1m_tokens=0.150,
        output_price_per_1m_tokens=0.600,
    )

    return start_mining(args, llm, cost_analyzer, rate_limiter)
